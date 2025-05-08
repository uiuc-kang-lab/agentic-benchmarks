#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

__global__ void kldiv_shared_memory_optimized_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    extern __shared__ float shared_mem[];
    float* shared_logs = shared_mem;
    float* shared_targets = &shared_mem[blockDim.x];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Load data into shared memory
    if (idx < n) {
        shared_logs[tid] = log_predictions[idx];
        shared_targets[tid] = targets[idx];
    }
    __syncthreads();

    // Compute KL divergence using shared memory
    if (idx < n) {
        sum = expf(shared_logs[tid]) - shared_targets[tid] * shared_logs[tid];
    }

    // Warp-level reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Block-level reduction
    if (tid % warpSize == 0) {
        atomicAdd(output, sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const size_t shared_mem_size = 2 * threads * sizeof(float);

    kldiv_shared_memory_optimized_kernel<<<blocks, threads, shared_mem_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence with shared memory optimization");
}
