#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    __shared__ float smem[THREADS_PER_BLOCK];
    
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int batch_idx = blockIdx.x;
    
    const float* row_x = x + batch_idx * num_features;
    float* row_y = y + batch_idx * num_features;
    
    // Initialize for max reduction
    float thread_max = -1e20;
    
    // Vectorized load and max computation
    #pragma unroll 4
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        thread_max = max(thread_max, __ldg(row_x + i));
    }
    
    // Warp-level reduction (no divergent branches)
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        float other = __shfl_down_sync(0xffffffff, thread_max, offset);
        thread_max = max(thread_max, other);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        smem[warp_id] = thread_max;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (tid < (THREADS_PER_BLOCK / WARP_SIZE)) {
        thread_max = smem[tid];
    }
    
    if (tid < WARP_SIZE) {
        #pragma unroll
        for (int offset = (THREADS_PER_BLOCK/WARP_SIZE)/2; offset > 0; offset >>= 1) {
            float other = __shfl_down_sync(0xffffffff, thread_max, offset);
            thread_max = max(thread_max, other);
        }
    }
    
    // Broadcast max value to all threads
    float max_val = __shfl_sync(0xffffffff, thread_max, 0);
    __syncthreads();
    
    // Compute exp and sum
    float thread_sum = 0.0f;
    
    #pragma unroll 4
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        float val = __expf(__ldg(row_x + i) - max_val);
        row_y[i] = val;  // Store intermediate result
        thread_sum += val;
    }
    
    // Warp-level sum reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    if (lane_id == 0) {
        smem[warp_id] = thread_sum;
    }
    __syncthreads();
    
    if (tid < (THREADS_PER_BLOCK / WARP_SIZE)) {
        thread_sum = smem[tid];
    }
    
    if (tid < WARP_SIZE) {
        #pragma unroll
        for (int offset = (THREADS_PER_BLOCK/WARP_SIZE)/2; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
    }
    
    float sum_val = __shfl_sync(0xffffffff, thread_sum, 0);
    
    // Normalize with vectorized operations
    #pragma unroll 4
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        row_y[i] /= sum_val;
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