#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <limits>
#include <c10/cuda/CUDAStream.h>

// Utility function for warp-level min reduction using shuffle primitive
template <typename scalar_t>
__inline__ __device__ scalar_t warpReduceMin(scalar_t val) {
    // Use full mask for active threads in the warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        scalar_t temp = __shfl_down_sync(0xffffffff, val, offset);
        val = (temp < val) ? temp : val;
    }
    return val;
}

// Kernel: Each warp computes the min reduction for one output element using only warp-level primitives
// The input tensor is logically reshaped as [outer, r, inner] and the reduction is performed along dimension r.
template <typename scalar_t>
__global__ void min_reduce_warp_only_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int outer,
    int r,
    int inner) {

    const int warpSize = 32;
    // Map each warp to one output element
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    if (warp_id >= outer * inner) return;

    int lane = threadIdx.x % warpSize;

    int outer_idx = warp_id / inner;
    int inner_idx = warp_id % inner;
    int base = outer_idx * (r * inner) + inner_idx;

    // Each warp thread computes a partial minimum over the r-dimension with stride = warpSize
    scalar_t local_min = std::numeric_limits<scalar_t>::max();
    for (int j = lane; j < r; j += warpSize) {
        int idx = base + j * inner;
        scalar_t val = input[idx];
        local_min = (val < local_min) ? val : local_min;
    }

    // Final warp-level reduction using __shfl_down_sync
    local_min = warpReduceMin<scalar_t>(local_min);

    // The first lane in the warp writes the output
    if (lane == 0) {
        output[warp_id] = local_min;
    }
}

// Forward function: sets up tensor dimensions and launches the kernel
torch::Tensor forward(torch::Tensor input, int64_t dim) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }

    int ndim = input.dim();
    TORCH_CHECK(dim >= 0 && dim < ndim, "dim out of range");

    // Calculate dimensions: outer dimensions, reduction dimension (r), and inner dimensions
    int outer = 1;
    for (int i = 0; i < dim; i++) {
        outer *= input.size(i);
    }
    int r = input.size(dim);
    int inner = 1;
    for (int i = dim + 1; i < ndim; i++) {
        inner *= input.size(i);
    }

    // Create output shape by removing the reduced dimension
    std::vector<int64_t> output_shape;
    for (int i = 0; i < ndim; i++) {
        if (i != dim) {
            output_shape.push_back(input.size(i));
        }
    }
    auto output = torch::empty(output_shape, input.options());

    // Each warp (32 threads) computes one output element
    int total_warps = outer * inner;
    const int threads_per_block = 128;  // 128 threads per block (4 warps per block)
    int num_blocks = (total_warps * warpSize + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_ALL_TYPES(input.scalar_type(), "min_reduce_warp_only_cuda", ([&] {
        min_reduce_warp_only_kernel<scalar_t><<<num_blocks, threads_per_block, 0,
            c10::cuda::getCurrentCUDAStream().stream()>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                outer,
                r,
                inner);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Min reduction using only warp-level primitives (CUDA)");
}
