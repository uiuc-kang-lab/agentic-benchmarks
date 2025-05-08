#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for KL divergence with coalesced memory accesses
__global__ void kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* output,
    const int n) {

    // Compute global thread index
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Use shared memory for reduction
    extern __shared__ float shared_data[];
    float local_sum = 0.0f;

    // Each thread processes elements in a coalesced manner
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        // Global memory accesses are coalesced since i increases consecutively
        float log_pred = log_predictions[i];
        float target = targets[i];
        local_sum += expf(log_pred) - target * log_pred;
    }

    // Store the local sum to shared memory
    shared_data[threadIdx.x] = local_sum;
    __syncthreads();

    // Intra-block reduction in shared memory
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread in the block writes the reduced sum to global memory using an atomic add
    if (threadIdx.x == 0) {
        atomicAdd(output, shared_data[0]);
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
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
