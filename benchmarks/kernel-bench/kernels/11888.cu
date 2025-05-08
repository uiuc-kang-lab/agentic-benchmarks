#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define warp size and block size
#define WARP_SIZE 32
#define BLOCK_SIZE 256

// Warp-level reduction using shuffle intrinsics
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Combined kernel using grid-stride loop for load balance and warp-level reduction
__global__ void kl_div_kernel_combined(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    float thread_sum = 0.0f;
    
    // Grid-stride loop to process all elements regardless of grid configuration
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        // KL divergence term: exp(log_pred) - target * log_pred
        thread_sum += expf(log_pred) - target * log_pred;
    }

    // Perform warp-level reduction using shuffle intrinsics
    thread_sum = warp_reduce(thread_sum);

    // Allocate shared memory to store per-warp results
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;

    // Only lane 0 of each warp writes its result to shared memory
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // Let the first warp reduce the per-warp sums stored in shared memory
    if (warp_id == 0) {
        // Each thread in the first warp loads one value from shared memory if available
        float block_sum = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce(block_sum);
        if (lane == 0) {
            // Accumulate block result into global output using atomic addition
            atomicAdd(output, block_sum);
        }
    }
}

// Host function to launch the kernel
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    // Create output tensor (single element) initialized to zero
    auto output = torch::zeros({1}, log_predictions.options());

    // Launch configuration
    const int threads = BLOCK_SIZE;
    // The number of blocks is computed to cover all data; grid-stride ensures full coverage
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = (threads / WARP_SIZE) * sizeof(float);

    kl_div_kernel_combined<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Normalize the result by dividing by n and return
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA combined efficient kernel)");
}
