#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

// CUDA kernel using warp-level primitives for reduction
// This version replaces shared memory operations with warp-level primitives
// to optimize performance for small reductions.

template <typename scalar_t>
__global__ void argmin_warp_optimized_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane = tid % 32;
    const int num_warps = blockDim.x / 32;
    
    int64_t slice_idx = blockIdx.x;
    if (slice_idx >= outer_size * inner_size) return;
    
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;
    
    scalar_t local_min = FLT_MAX;
    int local_min_idx = 0;
    
    for (int k = lane; k < K; k += 32) {
        scalar_t val = x[outer * (K * inner_size) + k * inner_size + inner];
        if (val < local_min) {
            local_min = val;
            local_min_idx = k;
        }
    }
    
    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t other_min = __shfl_down_sync(0xFFFFFFFF, local_min, offset);
        int other_idx = __shfl_down_sync(0xFFFFFFFF, local_min_idx, offset);
        if (other_min < local_min) {
            local_min = other_min;
            local_min_idx = other_idx;
        }
    }
    
    if (lane == 0) {
        // Store result of each warp
        atomicMin(&output[slice_idx], local_min_idx);
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
    
    int threads = 256;
    int blocks = outer_size * inner_size;
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_warp_optimized_kernel<scalar_t><<<blocks, threads>>>(
            x_data, output_data, K, outer_size, inner_size);
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
