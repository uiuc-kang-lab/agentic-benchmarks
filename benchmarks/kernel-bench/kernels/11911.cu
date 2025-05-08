#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define ELEMENTS_PER_THREAD 4

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_variable_block(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int global_thread_id = blockIdx.x * blockDim.x + tid;
    const int num_threads = gridDim.x * blockDim.x;
    
    extern __shared__ float warp_results[];
    
    float thread_sum = 0.0f;
    
    // Distribute workload evenly across all threads
    for (int idx = global_thread_id; idx < n; idx += num_threads) {
        const float log_pred = log_predictions[idx];
        const float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
    }
    
    // First level reduction within warps
    thread_sum = warp_reduce(thread_sum);
    
    // Store warp results only once per warp
    if (tid % WARP_SIZE == 0) {
        warp_results[tid / WARP_SIZE] = thread_sum;
    }
    
    // Synchronize only when necessary
    __syncthreads();
    
    // Final reduction across warps
    if (tid < (blockDim.x / WARP_SIZE)) {
        float warp_sum = warp_results[tid];
        warp_sum = warp_reduce(warp_sum);
        
        if (tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Experiment with different block sizes
    const int block_sizes[] = {32, 64, 128, 256, 512};
    int optimal_block_size = 256;  // Default to 256
    float min_time = FLT_MAX;

    for (int block_size : block_sizes) {
        const int blocks = (n + block_size * ELEMENTS_PER_THREAD - 1) / (block_size * ELEMENTS_PER_THREAD);
        const int warps_per_block = block_size / WARP_SIZE;
        const int shared_mem = warps_per_block * sizeof(float);

        // Measure execution time
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        kl_div_kernel_variable_block<<<blocks, block_size, shared_mem>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n
        );

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (milliseconds < min_time) {
            min_time = milliseconds;
            optimal_block_size = block_size;
        }

        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Launch kernel with optimal block size
    const int blocks = (n + optimal_block_size * ELEMENTS_PER_THREAD - 1) / (optimal_block_size * ELEMENTS_PER_THREAD);
    const int warps_per_block = optimal_block_size / WARP_SIZE;
    const int shared_mem = warps_per_block * sizeof(float);

    kl_div_kernel_variable_block<<<blocks, optimal_block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA variable block size)");
}