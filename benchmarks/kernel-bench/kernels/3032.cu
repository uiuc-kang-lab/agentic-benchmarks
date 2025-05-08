#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;
    
    // Each thread maintains its local max
    float thread_max = -INFINITY;
    
    // Process elements with minimal divergence
    #pragma unroll 4
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        thread_max = max(thread_max, x_row[i]);
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        float other_max = __shfl_down_sync(0xffffffff, thread_max, offset);
        thread_max = max(thread_max, other_max);
    }
    
    // First thread in each warp writes to shared memory
    __shared__ float warp_maxes[THREADS_PER_BLOCK/WARP_SIZE];
    if (lane_id == 0) {
        warp_maxes[warp_id] = thread_max;
    }
    __syncthreads();
    
    // First warp reduces the warp maxes
    if (warp_id == 0 && lane_id < (THREADS_PER_BLOCK/WARP_SIZE)) {
        float warp_max = warp_maxes[lane_id];
        #pragma unroll
        for (int offset = (THREADS_PER_BLOCK/WARP_SIZE)/2; offset > 0; offset /= 2) {
            float other_max = __shfl_down_sync(0xffffffff, warp_max, offset);
            warp_max = max(warp_max, other_max);
        }
        if (lane_id == 0) {
            warp_maxes[0] = warp_max;
        }
    }
    __syncthreads();
    
    float max_val = warp_maxes[0];
    
    // Compute exponentials and partial sums
    float thread_sum = 0.0f;
    #pragma unroll 4
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        float val = __expf(x_row[i] - max_val);
        y_row[i] = val;
        thread_sum += val;
    }
    
    // Warp-level reduction for sum using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // First thread in each warp writes to shared memory
    __shared__ float warp_sums[THREADS_PER_BLOCK/WARP_SIZE];
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();
    
    // First warp reduces the warp sums
    if (warp_id == 0 && lane_id < (THREADS_PER_BLOCK/WARP_SIZE)) {
        float warp_sum = warp_sums[lane_id];
        #pragma unroll
        for (int offset = (THREADS_PER_BLOCK/WARP_SIZE)/2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (lane_id == 0) {
            warp_sums[0] = warp_sum;
        }
    }
    __syncthreads();
    
    float sum_val = warp_sums[0];
    
    // Normalize with coalesced memory access
    #pragma unroll 4
    for (int i = tid; i < num_features; i += THREADS_PER_BLOCK) {
        y_row[i] = y_row[i] / sum_val;
    }
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 block_dim(THREADS_PER_BLOCK);
    dim3 grid_dim(batch_size);
    
    softmax_kernel<<<grid_dim, block_dim>>>(x, y, num_features);
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor.");
    TORCH_CHECK(x.dim() == 2, "Input tensor must be 2D.");
    TORCH_CHECK(x.scalar_type() == torch::kFloat32, "Input tensor must be float32.");
    
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