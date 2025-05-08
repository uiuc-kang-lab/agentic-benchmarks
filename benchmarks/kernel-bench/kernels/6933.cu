#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>

template <typename scalar_t>
__global__ void optimized_argmin_kernel(
    const scalar_t* __restrict__ x,
    int64_t* __restrict__ output,
    int K,
    int64_t outer_size,
    int64_t inner_size) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int block_size = blockDim.x;
    
    __shared__ scalar_t s_min_vals[256];
    __shared__ int s_min_indices[256];
    
    int64_t slice_idx = bid;
    if (slice_idx >= outer_size * inner_size) return;
    
    int64_t outer = slice_idx / inner_size;
    int64_t inner = slice_idx % inner_size;
    
    scalar_t local_min = FLT_MAX;
    int local_min_idx = 0;
    
    #pragma unroll 4
    for (int k = tid; k < K; k += block_size) {
        scalar_t val = __ldg(&x[outer * (K * inner_size) + k * inner_size + inner]);
        if (val < local_min) {
            local_min = val;
            local_min_idx = k;
        }
    }
    
    s_min_vals[tid] = local_min;
    s_min_indices[tid] = local_min_idx;
    __syncthreads();
    
    for (int stride = block_size/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            if (s_min_vals[tid + stride] < s_min_vals[tid]) {
                s_min_vals[tid] = s_min_vals[tid + stride];
                s_min_indices[tid] = s_min_indices[tid + stride];
            }
        }
        __syncthreads();
    }
    
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
    
    if (tid == 0) {
        output[slice_idx] = s_min_indices[0];
    }
}

at::Tensor argmin_cuda_forward(const at::Tensor &x, int64_t dim) {
    auto stream = at::cuda::getCurrentCUDAStream();
    
    int dims = x.dim();
    if (dim < 0) dim += dims;
    TORCH_CHECK(dim >= 0 && dim < dims, "Reduction dim out of range");
    
    int64_t outer_size = 1;
    for (int i = 0; i < dim; i++) outer_size *= x.size(i);
    int K = x.size(dim);
    int64_t inner_size = 1;
    for (int i = dim + 1; i < dims; i++) inner_size *= x.size(i);
    
    auto output = at::empty({outer_size, inner_size}, x.options().dtype(at::kLong));
    
    const int threads = 256;
    const int blocks = outer_size * inner_size;
    
    AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::Half, x.scalar_type(), "argmin_cuda_forward", ([&] {
        optimized_argmin_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            output.data_ptr<int64_t>(),
            K, outer_size, inner_size);
    }));
    
    return output.reshape(out_sizes);
}