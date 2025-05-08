#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__constant__ float const_log_predictions[1024];

__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

__device__ __forceinline__ void warp_reduce(volatile float* sdata, unsigned int tid) {
    if (blockDim.x >= 64) sdata[tid] += sdata[tid + 32];
    if (blockDim.x >= 32) sdata[tid] += sdata[tid + 16];
    if (blockDim.x >= 16) sdata[tid] += sdata[tid + 8];
    if (blockDim.x >= 8) sdata[tid] += sdata[tid + 4];
    if (blockDim.x >= 4) sdata[tid] += sdata[tid + 2];
    if (blockDim.x >= 2) sdata[tid] += sdata[tid + 1];
}

__device__ __forceinline__ void block_reduce(float* partial_sums, unsigned int tid) {
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            partial_sums[tid] += partial_sums[tid + stride];
        }
        __syncthreads();
    }
    if (tid < 32) warp_reduce(partial_sums, tid);
}

__global__ void kl_div_kernel_combined(
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    
    const unsigned int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    
    extern __shared__ float partial_sums[];
    float sum = 0.0f;
    while (idx < n) {
        float log_pred = const_log_predictions[idx];
        sum += compute_kl_element(log_pred, targets[idx]);
        idx += stride;
    }
    partial_sums[tid] = sum;
    __syncthreads();
    block_reduce(partial_sums, tid);
    if (tid == 0) {
        atomicAdd(output, partial_sums[0]);
    }
}

torch::Tensor kl_div_cuda_forward_combined(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    cudaMemcpyToSymbol(const_log_predictions, log_predictions.data_ptr<float>(), n * sizeof(float));
    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);
    kl_div_kernel_combined<<<blocks, threads, shared_mem>>>(
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_combined, "KL divergence forward (CUDA)");
}