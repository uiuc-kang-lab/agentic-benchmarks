#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel for KL divergence with overlap of computation and memory transfers
__global__ void streamed_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    // Simplified indexing for the kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    extern __shared__ float partial_sums[];
    float sum = 0.0f;
    
    for (int i = idx; i < n; i += stride) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }
    
    partial_sums[threadIdx.x] = sum;
    __syncthreads();
    
    // Reduce within block
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + offset];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

// Host function leveraging CUDA streams
torch::Tensor streamed_kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = threads * sizeof(float);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate device memory in the stream
    float *d_log_predictions, *d_targets, *d_output;
    cudaMalloc((void**)&d_log_predictions, n * sizeof(float));
    cudaMalloc((void**)&d_targets, n * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));
    
    // Asynchronously copy data to device
    cudaMemcpyAsync(d_log_predictions, log_predictions.data_ptr<float>(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_targets, targets.data_ptr<float>(), n * sizeof(float), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_output, output.data_ptr<float>(), sizeof(float), cudaMemcpyHostToDevice, stream);
    
    // Launch kernel in stream
    streamed_kl_div_kernel<<<blocks, threads, shared_mem, stream>>>(
        d_log_predictions,
        d_targets,
        d_output,
        n
    );
    
    // Asynchronously copy the result back
    cudaMemcpyAsync(output.data_ptr<float>(), d_output, sizeof(float), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    // Free memory
    cudaFree(d_log_predictions);
    cudaFree(d_targets);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &streamed_kl_div_cuda_forward, "KL divergence with stream optimization (CUDA)");
}
