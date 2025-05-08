#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses stride loops to handle large workloads with correct boundary handling
__global__ void kldiv_stride_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int64_t n) {

    // Compute global thread index and total stride
    int64_t global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    // Process elements using a stride loop
    for (int64_t i = global_idx; i < n; i += stride) {
        float lp = log_predictions[i];
        float tgt = targets[i];
        local_sum += expf(lp) - tgt * lp;
    }

    // Reduce the per-thread sum within the block using shared memory
    extern __shared__ float shared_sum[];
    shared_sum[threadIdx.x] = local_sum;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread in the block atomically adds the block result to the global output
    if (threadIdx.x == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}

// CUDA forward function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {

    const int64_t n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    size_t shared_memory = threads * sizeof(float);

    kldiv_stride_kernel<<<blocks, threads, shared_memory>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA, stride loop)");
}
