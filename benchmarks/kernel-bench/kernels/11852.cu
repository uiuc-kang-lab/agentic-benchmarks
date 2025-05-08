#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kldiv_atomic_optimized_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    extern __shared__ float sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    float sum = 0.0f;

    // Loop over the data, stride by grid size
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += gridSize;
    }

    // Each thread writes its local sum into shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Only one thread per block writes to global output
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const unsigned int threads = 256;
    const unsigned int blocks = (n + threads - 1) / threads;
    const size_t shared_mem = threads * sizeof(float);

    kldiv_atomic_optimized_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence atomic optimized forward");
}