#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Warp-level reduction function using shuffle operations
template <typename scalar_t>
__device__ inline scalar_t warpReduceMin(scalar_t val) {
  for (int offset = 16; offset > 0; offset /= 2) {
    val = min(val, __shfl_down_sync(0xffffffff, val, offset));
  }
  return val;
}

// Kernel for warp-optimized min reduction over a specified dimension
// The input tensor is logically reshaped into [outer, r, inner], and reduction is over the r dimension

template <typename scalar_t>
__global__ void warp_optimized_min_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int outer,
    const int r,
    const int inner) {

    int idx = blockIdx.x;
    if (idx >= outer * inner) return;

    int outer_idx = idx / inner;
    int inner_idx = idx % inner;
    const scalar_t* in_ptr = input + outer_idx * (r * inner) + inner_idx;

    // Each thread computes partial minimum
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    for (int j = threadIdx.x; j < r; j += blockDim.x) {
        scalar_t val = in_ptr[j * inner];
        local_min = min(val, local_min);
    }

    // Perform warp-level reduction
    local_min = warpReduceMin(local_min);

    // Write the result from the first thread of the warp
    if (threadIdx.x % 32 == 0) {
        atomicMin(&output[idx], local_min);
    }
}

// Host forward function
// Reshapes input into dimensions [outer, r, inner] according to the reduction dimension and launches the kernel
torch::Tensor forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int ndim = input.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    int outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= input.size(i);
    }
    int r = input.size(dim);
    int inner = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner *= input.size(i);
    }

    std::vector<int64_t> output_shape;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            output_shape.push_back(input.size(i));
        }
    }

    auto output = torch::full(output_shape, std::numeric_limits<float>::max(), input.options());
    int total = outer * inner;

    int threads = 256; // Using 256 threads per block
    int blocks = total;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "warp_optimized_min_reduce_cuda", ([&] {
        warp_optimized_min_reduction_kernel<scalar_t><<<blocks, threads, 0, c10::cuda::getCurrentCUDAStream().stream()>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), outer, r, inner);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized min reduction (CUDA)");
}
