#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void coalesced_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int block_offset = blockIdx.x * blockDim.x;
    const int idx = block_offset + tid;
    const int stride = blockDim.x * gridDim.x;
    
    extern __shared__ float partial_sums[];
    float sum = 0.0f;

    // Coalesced memory access
    for (int i = idx; i < n; i += stride) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }

    partial_sums[tid] = sum;
    __syncthreads();
    
    // Reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            partial_sums[tid] += partial_sums[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;  // Multiple of warp size
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);
    
    coalesced_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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