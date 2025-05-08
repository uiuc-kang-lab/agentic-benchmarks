#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Maximum number of elements that can be stored in constant memory
// For float, 16384 elements require 64KB (since 16384*4 = 65536 bytes)
#define MAX_CONST_ELEMENTS 16384

// Define constant memory for frequently-read data
__constant__ float c_log_predictions[MAX_CONST_ELEMENTS];
__constant__ float c_targets[MAX_CONST_ELEMENTS];

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

// Warp-level reduction using shuffle operations
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel that uses constant memory for input data
__global__ void kl_div_kernel_const(
    float* __restrict__ output,
    const int n,
    const int elements_per_thread) {

    int tid = threadIdx.x;
    int wid = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    int global_thread_id = blockIdx.x * blockDim.x + tid;

    extern __shared__ float warp_results[];
    float thread_sum = 0.0f;

    // Each thread processes 'elements_per_thread' consecutive elements using grid-stride loop
    int start_idx = global_thread_id * elements_per_thread;
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        int idx = start_idx + i;
        if (idx < n) {
            // Load from constant memory
            float log_pred = c_log_predictions[idx];
            float target = c_targets[idx];
            thread_sum += expf(log_pred) - target * log_pred;
        }
    }

    // Intra-warp reduction using shuffle
    thread_sum = warp_reduce(thread_sum);

    // Each warp's leader writes its result to shared memory
    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }
    __syncthreads();

    // Final reduction across warps in the block
    if (wid == 0) {
        float block_sum = (lane < (blockDim.x / WARP_SIZE)) ? warp_results[lane] : 0.0f;
        block_sum = warp_reduce(block_sum);
        if (lane == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host function called from PyTorch
// Note: It is assumed that the number of elements in the input tensor
// does not exceed MAX_CONST_ELEMENTS. If n is larger, this kernel
// may not be applicable without further modifications.

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Copy input data to constant memory. We assume n <= MAX_CONST_ELEMENTS.
    cudaMemcpyToSymbol(c_log_predictions, log_predictions.data_ptr<float>(), n * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    cudaMemcpyToSymbol(c_targets, targets.data_ptr<float>(), n * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    // Determine grid dimensions
    const int elements_per_thread = ELEMENTS_PER_THREAD;
    const int total_threads = (n + elements_per_thread - 1) / elements_per_thread;
    const int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Shared memory for warp-level partial sums
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int shared_mem = warps_per_block * sizeof(float);

    // Launch kernel
    kl_div_kernel_const<<<blocks, BLOCK_SIZE, shared_mem>>>(
        output.data_ptr<float>(),
        n,
        elements_per_thread
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA constant memory optimized)");
}
