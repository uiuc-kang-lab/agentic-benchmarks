#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define NUM_WARPS (THREADS_PER_BLOCK / WARP_SIZE)

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Calculate work distribution per warp
    const int items_per_warp = (num_features + NUM_WARPS - 1) / NUM_WARPS;
    const int warp_start = warp_id * items_per_warp;
    const int warp_end = min(warp_start + items_per_warp, num_features);
    
    // Find maximum value within warp's segment
    float thread_max = -INFINITY;
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        thread_max = max(thread_max, x[batch_idx * num_features + i]);
    }
    
    // Warp-level reduction for maximum using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    
    // Broadcast warp max to all threads in warp
    float warp_max = __shfl_sync(0xffffffff, thread_max, 0);
    
    // First thread in each warp participates in final reduction
    if (lane == 0) {
        atomicMax((int*)&y[batch_idx * num_features], __float_as_int(warp_max));
    }
    __syncthreads();
    
    // Get final max value
    float max_val = __int_as_float(y[batch_idx * num_features]);
    
    // Compute exponentials and sum
    float thread_sum = 0.0f;
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        const float val = __expf(x[batch_idx * num_features + i] - max_val);
        y[batch_idx * num_features + i] = val;
        thread_sum += val;
    }
    
    // Warp-level reduction for sum
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp adds to global sum
    if (lane == 0) {
        atomicAdd(&y[batch_idx * num_features + num_features - 1], thread_sum);
    }
    __syncthreads();
    
    // Get final sum
    float sum_val = y[batch_idx * num_features + num_features - 1];
    
    // Normalize
    for (int i = warp_start + lane; i < warp_end; i += WARP_SIZE) {
        y[batch_idx * num_features + i] /= sum_val;
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