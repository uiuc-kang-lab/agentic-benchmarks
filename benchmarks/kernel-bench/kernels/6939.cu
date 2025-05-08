#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void argmin_tuned_blocks_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;  // Using 128 threads per block
    
    // Shared memory for partial results - sized for 128 threads
    __shared__ scalar_t s_min_vals[128];
    __shared__ int s_min_indices[128];
    
    // Calculate which slice this block is processing
    int64_t slice_idx = bid;
    if (slice_idx >= outer_size * inner_size) return;
    
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;
    
    // Initialize with maximum value
    scalar_t local_min = FLT_MAX;
    int local_min_idx = 0;
    
    // Each thread processes elements strided by block_size
    // Using __ldg for cached memory access
    for (int k = tid; k < K; k += block_size) {
        scalar_t val = __ldg(&x[outer * (K * inner_size) + k * inner_size + inner]);
        if (val < local_min) {
            local_min = val;
            local_min_idx = k;
        }
    }
    
    // Store in shared memory
    s_min_vals[tid] = local_min;
    s_min_indices[tid] = local_min_idx;
    __syncthreads();
    
    // Reduce within the block - optimized for 128 threads
    // Unrolled first iteration for better instruction scheduling
    if (tid < 64) {
        if (s_min_vals[tid + 64] < s_min_vals[tid]) {
            s_min_vals[tid] = s_min_vals[tid + 64];
            s_min_indices[tid] = s_min_indices[tid + 64];
        }
    }
    __syncthreads();
    
    // Continue reduction with remaining iterations
    if (tid < 32) {
        if (s_min_vals[tid + 32] < s_min_vals[tid]) {
            s_min_vals[tid] = s_min_vals[tid + 32];
            s_min_indices[tid] = s_min_indices[tid + 32];
        }
        __syncwarp();
        
        if (s_min_vals[tid + 16] < s_min_vals[tid]) {
            s_min_vals[tid] = s_min_vals[tid + 16];
            s_min_indices[tid] = s_min_indices[tid + 16];
        }
        __syncwarp();
        
        if (s_min_vals[tid + 8] < s_min_vals[tid]) {
            s_min_vals[tid] = s_min_vals[tid + 8];
            s_min_indices[tid] = s_min_indices[tid + 8];
        }
        __syncwarp();
        
        if (s_min_vals[tid + 4] < s_min_vals[tid]) {
            s_min_vals[tid] = s_min_vals[tid + 4];
            s_min_indices[tid] = s_min_indices[tid + 4];
        }
        __syncwarp();
        
        if (s_min_vals[tid + 2] < s_min_vals[tid]) {
            s_min_vals[tid] = s_min_vals[tid + 2];
            s_min_indices[tid] = s_min_indices[tid + 2];
        }
        __syncwarp();
        
        if (s_min_vals[tid + 1] < s_min_vals[tid]) {
            s_min_vals[tid] = s_min_vals[tid + 1];
            s_min_indices[tid] = s_min_indices[tid + 1];
        }
    }
    
    // Write result
    if (tid == 0) {
        output[slice_idx] = s_min_indices[0];
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
    
    // Using 128 threads per block instead of 256
    int threads = 128;
    int blocks = outer_size * inner_size;
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        const scalar_t* x_data = x.data_ptr<scalar_t>();
        int64_t* output_data = output.data_ptr<int64_t>();
        argmin_tuned_blocks_kernel<scalar_t><<<blocks, threads>>>(
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