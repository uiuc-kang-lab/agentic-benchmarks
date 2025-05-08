#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ x, 
                             float* __restrict__ y, 
                             const int num_features) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warps_per_block = THREADS_PER_BLOCK / WARP_SIZE;
    
    // Calculate optimal stride based on feature size
    const int stride = THREADS_PER_BLOCK;
    
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;
    
    __shared__ float warp_maxes[warps_per_block];
    __shared__ float warp_sums[warps_per_block];

    // Find max value
    float thread_max = -INFINITY;
    #pragma unroll 4
    for (int i = tid; i < num_features; i += stride) {
        if (i < num_features) {
            thread_max = max(thread_max, __ldg(&x_row[i]));
        }
    }
    
    // Warp-level reduction
    float warp_max = warp_reduce_max(thread_max);
    
    if (lane_id == 0) {
        warp_maxes[warp_id] = warp_max;
    }
    __syncthreads();
    
    // Final reduction for max
    if (tid < warps_per_block) {
        float block_max = warp_maxes[tid];
        block_max = warp_reduce_max(block_max);
        if (tid == 0) {
            warp_maxes[0] = block_max;
        }
    }
    __syncthreads();
    
    const float max_val = warp_maxes[0];
    
    // Compute exp and sum
    float thread_sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < num_features; i += stride) {
        if (i < num_features) {
            const float val = __expf(__ldg(&x_row[i]) - max_val);
            y_row[i] = val;
            thread_sum += val;
        }
    }
    
    // Warp-level reduction for sum
    float warp_sum = warp_reduce_sum(thread_sum);
    
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction for sum
    if (tid < warps_per_block) {
        float block_sum = warp_sums[tid];
        block_sum = warp_reduce_sum(block_sum);
        if (tid == 0) {
            warp_sums[0] = block_sum;
        }
    }
    __syncthreads();
    
    const float sum_val = warp_sums[0];
    
    // Normalize with optimized stride
    #pragma unroll 4
    for (int i = tid; i < num_features; i += stride) {
        if (i < num_features) {
            y_row[i] /= sum_val;
        }
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size);
    dim3 block(THREADS_PER_BLOCK);
    
    softmax_kernel<<<grid, block>>>(x, y, num_features);
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32");
    
    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(), x.size(0), x.size(1));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}