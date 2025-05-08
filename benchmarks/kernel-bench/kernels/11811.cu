#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Shared memory based KL-Divergence kernel
__global__ void kl_div_kernel_shared_memory(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Declare shared memory
    extern __shared__ float shared_mem[];

    float local_sum = 0.0f;
    for (int idx = gid; idx < n; idx += stride) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        local_sum += __expf(log_pred) - target * log_pred;
    }

    shared_mem[tid] = local_sum;
    __syncthreads();

    // Reduce within block
    for(int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_mem[tid] += shared_mem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, shared_mem[0]);
    }
}

torch::Tensor kl_div_cuda_forward_shared_memory(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);

    kl_div_kernel_shared_memory<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_shared_memory, "KL divergence forward with shared memory optimization (CUDA)");
}