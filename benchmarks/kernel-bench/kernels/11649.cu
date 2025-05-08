#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float const_log_predictions[1024];

// CUDA kernel for KL divergence calculation with adaptive memory usage
__global__ void adaptive_kl_div_kernel(
    const float* log_predictions,
    const float* targets, 
    float* output,
    const int n,
    const bool use_const_mem) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float partial_sums[];
    
    float sum = 0.0f;
    
    while (idx < n) {
        float log_pred;
        if (use_const_mem) {
            log_pred = const_log_predictions[idx];
        } else {
            log_pred = log_predictions[idx];
        }
        float target = targets[idx];
        sum += __expf(log_pred) - target * log_pred;  // Using faster intrinsic
        
        idx += blockDim.x * gridDim.x;
    }
    
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    
    // Warp-level reduction first
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Block-level reduction for the first thread in each warp
    if (threadIdx.x % warpSize == 0) {
        partial_sums[threadIdx.x / warpSize] = sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (threadIdx.x < blockDim.x / warpSize) {
        sum = partial_sums[threadIdx.x];
        #pragma unroll
        for (int offset = (blockDim.x/warpSize)/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor adaptive_kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    bool use_const_mem = (n <= 1024);
    if (use_const_mem) {
        cudaMemcpyToSymbol(const_log_predictions, 
                          log_predictions.data_ptr<float>(), 
                          n * sizeof(float));
    }
    
    adaptive_kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        use_const_mem
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_kl_div_cuda_forward, "Adaptive KL divergence forward (CUDA)");
}