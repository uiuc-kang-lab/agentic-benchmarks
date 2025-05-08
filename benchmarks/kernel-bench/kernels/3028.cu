#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    
    __shared__ float s_max[THREADS_PER_BLOCK / WARP_SIZE];
    __shared__ float s_sum[THREADS_PER_BLOCK / WARP_SIZE];
    
    const float* x_row = x + bid * num_features;
    float* y_row = y + bid * num_features;
    
    float thread_max = -INFINITY;
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        thread_max = max(thread_max, x_row[i]);
    }
    
    float warp_max = warp_reduce_max(thread_max);
    
    if (lane_id == 0) {
        s_max[warp_id] = warp_max;
    }
    __syncthreads();
    
    if (warp_id == 0 && lane_id < num_warps) {
        float block_max = s_max[lane_id];
        block_max = warp_reduce_max(block_max);
        s_max[0] = block_max;
    }
    __syncthreads();
    
    const float max_val = s_max[0];
    
    float thread_sum = 0.0f;
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        const float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        thread_sum += exp_val;
    }
    
    float warp_sum = warp_reduce_sum(thread_sum);
    
    if (lane_id == 0) {
        s_sum[warp_id] = warp_sum;
    }
    __syncthreads();
    
    if (warp_id == 0 && lane_id < num_warps) {
        float block_sum = s_sum[lane_id];
        block_sum = warp_reduce_sum(block_sum);
        s_sum[0] = block_sum;
    }
    __syncthreads();
    
    const float sum_val = s_sum[0];
    
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        y_row[i] /= sum_val;
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
    softmax_forward_cuda(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        x.size(0),
        x.size(1)
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (CUDA)");
}