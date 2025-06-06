#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// CUDA kernel using warp-level primitives for reduction
template <typename scalar_t>
__global__ void argmin_warp_kernel(const scalar_t* __restrict__ x,
                                  int64_t* __restrict__ output,
                                  int K,
                                  int64_t outer_size,
                                  int64_t inner_size) {
    const unsigned FULL_WARP = 0xffffffff;
    const int warp_size = 32;
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total_slices = outer_size * inner_size;
    if (idx >= total_slices) return;

    int64_t outer = idx / inner_size;
    int64_t inner = idx % inner_size;
    const scalar_t* slice_start = x + outer * (static_cast<int64_t>(K) * inner_size) + inner;

    // Initialize with the first value
    scalar_t min_val = slice_start[0];
    int min_index = 0;

    // Process K elements
    for (int k = threadIdx.x % warp_size; k < K; k += warp_size) {
        scalar_t val = slice_start[k * inner_size];
        if (val < min_val) {
            min_val = val;
            min_index = k;
        }
    }

    // Warp-level reduction using shuffle operations
    #pragma unroll
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        scalar_t other_val = __shfl_down_sync(FULL_WARP, min_val, offset);
        int other_index = __shfl_down_sync(FULL_WARP, min_index, offset);
        if (other_val < min_val) {
            min_val = other_val;
            min_index = other_index;
        }
    }

    // First thread in each warp writes the result
    if (threadIdx.x % warp_size == 0) {
        output[idx] = min_index;
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");

    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");

    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) {
        outer_size *= x.size(i);
    }
    int K = static_cast<int>(x.size(dim));
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) {
        inner_size *= x.size(i);
    }

    std::vector<int64_t> out_sizes;
    for (int i = 0; i < dims; i++) {
        if (i == dim) continue;
        out_sizes.push_back(x.size(i));
    }
    auto output = at::empty(out_sizes, x.options().dtype(at::kLong));

    int64_t total_slices = outer_size * inner_size;
    int threads = 256;
    int blocks = (total_slices + threads - 1) / threads;

    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_warp_kernel<scalar_t><<<blocks, threads>>>(x_data, output_data, K, outer_size, inner_size);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA kernel failed: ") + cudaGetErrorString(err));
    }
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &argmin_cuda_forward, "Argmin forward (CUDA)");
}