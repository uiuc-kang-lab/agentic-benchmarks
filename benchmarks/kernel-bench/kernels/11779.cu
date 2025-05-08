#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

// Kernel that uses shared memory to cache log_predictions and targets for reduced global memory access
__global__ void shared_memory_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    extern __shared__ float shared_mem[];
    float* shared_log_predictions = shared_mem;
    float* shared_targets = shared_mem + blockDim.x;

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int total_threads = gridDim.x * blockDim.x;
    float sum = 0.0f;

    // Load data into shared memory
    for (int i = tid; i < n; i += total_threads) {
        shared_log_predictions[threadIdx.x] = __ldg(log_predictions + i);
        shared_targets[threadIdx.x] = __ldg(targets + i);
        __syncthreads();

        // Compute KL divergence using shared memory
        float log_pred = shared_log_predictions[threadIdx.x];
        float target = shared_targets[threadIdx.x];
        sum += expf(log_pred) - target * log_pred;
        __syncthreads();
    }

    // Warp-level reduction using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Store warp results in shared memory
    if (threadIdx.x % WARP_SIZE == 0) {
        shared_mem[threadIdx.x / WARP_SIZE] = sum;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (threadIdx.x < (blockDim.x / WARP_SIZE)) {
        sum = shared_mem[threadIdx.x];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Host function to launch the kernel
torch::Tensor shared_memory_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem_size = 2 * threads * sizeof(float);

    shared_memory_kl_kernel<<<blocks, threads, shared_mem_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_kl_forward, "KL divergence with shared memory (CUDA)");
}