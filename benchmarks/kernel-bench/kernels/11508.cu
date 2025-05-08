#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(const float* __restrict__ log_predictions,
                             const float* __restrict__ targets, 
                             float* __restrict__ output,
                             const int n) {
    extern __shared__ float sdata[];

    // Calculate 1D global thread ID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float temp = 0.0f;

    // Process elements aligned for coalesced access
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        temp += expf(log_pred) - target * log_pred;
        idx += gridDim.x * blockDim.x;
    }

    // Store thread result in shared memory
    sdata[tid] = temp;
    __syncthreads();

    // Perform reduction within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) atomicAdd(output, sdata[0]);
}

torch::Tensor kl_div_cuda_forward(torch::Tensor log_predictions, torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    kl_div_kernel<<<blocks, threads, threads * sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward optimized (CUDA)");
}