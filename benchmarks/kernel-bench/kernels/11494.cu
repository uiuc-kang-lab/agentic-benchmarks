#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {
    
    int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    extern __shared__ float partial_sums[];
    
    float sum[4] = {0};

    // Process consecutive 4-element chunks with coalesced access
    for(int i = 0; i < 4; i++) {
        int idx = base_idx + i;
        if(idx < n) {
            float log_pred = log_predictions[idx];
            float target = targets[idx];
            sum[i] = expf(log_pred) - target * log_pred;
        }
    }

    // Register accumulation then shared memory store
    float total = sum[0] + sum[1] + sum[2] + sum[3];
    partial_sums[threadIdx.x] = total;
    __syncthreads();

    // Stable recursive reduction
    for(int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if(threadIdx.x < stride)
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride];
        __syncthreads();
    }

    if(threadIdx.x == 0)
        atomicAdd(output, partial_sums[0]);
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int elements_per_block = threads * 4;
    const int blocks = (n + elements_per_block - 1) / elements_per_block;
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