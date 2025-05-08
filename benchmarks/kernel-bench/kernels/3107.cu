#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#define THREADS_PER_BLOCK 256

template <unsigned int warp_size>
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = warp_size/2; offset > 0; offset /= 2) 
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

template <unsigned int warp_size>
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warp_size/2; offset > 0; offset /= 2) 
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void softmax_kernel(const float* __restrict__ x, float* __restrict__ y, int num_features) {
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane_id = tid % 32;
    const int warp_id = tid / 32;
    
    const float* x_row = x + batch_idx * num_features;
    float* y_row = y + batch_idx * num_features;
    
    extern __shared__ float sdata[];
    
    // Compute thread-local max
    float max_val = -INFINITY;
    for (int i = tid; i < num_features; i += blockDim.x)
        max_val = fmaxf(max_val, x_row[i]);
    
    // Warp-wise reduction for max
    max_val = warpReduceMax<32>(max_val);
    
    // Store warp results to shared memory
    if (lane_id == 0)
        sdata[warp_id] = max_val;
    
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        max_val = (tid < blockDim.x/32) ? sdata[lane_id] : -INFINITY;
        max_val = warpReduceMax<32>(max_val);
        // Broadcast the final max_val to shared memory for all threads to use
        if (lane_id == 0) {
            sdata[0] = max_val;
        }
    }
    __syncthreads();
    // Make sure all threads have the same max_val
    max_val = sdata[0];
    
    // Compute exponentials and sum
    float sum_val = 0.0f;
    for (int i = tid; i < num_features; i += blockDim.x) {
        float exp_val = __expf(x_row[i] - max_val);
        y_row[i] = exp_val;
        sum_val += exp_val;
    }
    
    // Warp-wise reduction for sum
    sum_val = warpReduceSum<32>(sum_val);
    
    // Store warp results to shared memory
    if (lane_id == 0)
        sdata[warp_id] = sum_val;
    
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0) {
        sum_val = (tid < blockDim.x/32) ? sdata[lane_id] : 0.0f;
        sum_val = warpReduceSum<32>(sum_val);
    }
    __syncthreads();
    
    // Normalize values
    for (int i = tid; i < num_features; i += blockDim.x)
        y_row[i] /= sum_val;
}

void softmax_forward_cuda(const float* x, float* y, int batch_size, int num_features) {
    dim3 grid(batch_size), block(THREADS_PER_BLOCK);
    int shared_mem_size = (THREADS_PER_BLOCK/32) * sizeof(float);
    softmax_kernel<<<grid, block, shared_mem_size>>>(x, y, num_features);
}

torch::Tensor forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "x must be CUDA tensor");
    auto y = torch::empty_like(x);
    softmax_forward_cuda(x.data_ptr<float>(), y.data_ptr<float>(),
                        x.size(0), x.size(1));
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softmax forward (optimized CUDA)");
}