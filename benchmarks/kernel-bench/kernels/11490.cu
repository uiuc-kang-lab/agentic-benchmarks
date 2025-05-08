#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    const int stride = blockDim.x * gridDim.x;
    int i = idx;
    
    // Use vectorized loads for better memory coalescing
    #pragma unroll 4
    while (i < n) {
        // Prefetch next iteration's data
        float4 log_pred_vec, target_vec;
        if (i + 3 < n) {
            log_pred_vec = *reinterpret_cast<const float4*>(&log_predictions[i]);
            target_vec = *reinterpret_cast<const float4*>(&targets[i]);
            
            // Process 4 elements at once
            sum += expf(log_pred_vec.x) - target_vec.x * log_pred_vec.x;
            sum += expf(log_pred_vec.y) - target_vec.y * log_pred_vec.y;
            sum += expf(log_pred_vec.z) - target_vec.z * log_pred_vec.z;
            sum += expf(log_pred_vec.w) - target_vec.w * log_pred_vec.w;
            
            i += 4;
        } else {
            // Handle remaining elements
            float log_pred = log_predictions[i];
            float target = targets[i];
            sum += expf(log_pred) - target * log_pred;
            i += 1;
        }
        i += stride - (i % stride);  // Align to next stride
    }
    
    partial_sums[threadIdx.x] = sum;
    __syncthreads();

    for (int s = blockDim.x/2; s >= 32; s >>= 1) {
        if (threadIdx.x < s) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + s];
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        float val = partial_sums[threadIdx.x];
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (threadIdx.x == 0)
            atomicAdd(output, val);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int sm_count = 132;  // H100 SMs
    const int max_blocks = sm_count * 4;  // Multiple blocks per SM
    int blocks = (n + threads - 1) / threads;
    blocks = min(max_blocks, blocks);
    
    const int shared_mem = threads * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}