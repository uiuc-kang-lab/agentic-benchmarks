#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = threadIdx.x;
    const int global_tid = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float shared_memory[];
    float* warp_sums = shared_memory;
    float* partial_sums = shared_memory + (BLOCK_SIZE / WARP_SIZE);

    float thread_sum = 0.0f;
    
    int idx = global_tid * ELEMENTS_PER_THREAD;
    #pragma unroll
    for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
        if (idx < n) {
            float log_pred = log_predictions[idx];
            float target = targets[idx];
            thread_sum += expf(log_pred) - target * log_pred;
            idx += BLOCK_SIZE * ELEMENTS_PER_THREAD;
        }
    }

    // Warp reduction
    thread_sum = warp_reduce(thread_sum);

    // First thread in warp writes to shared memory
    if ((tid % WARP_SIZE) == 0) { 
        warp_sums[tid / WARP_SIZE] = thread_sum;
    }
    __syncthreads();

    // Use one warp within the block to reduce the results from each warp
    float block_sum = 0.0f;
    if (tid < (BLOCK_SIZE / WARP_SIZE)) {
        block_sum = warp_sums[tid];
        block_sum = warp_reduce(block_sum);
    }

    if (tid == 0) {
        atomicAdd(output, block_sum);
    }
}


torch::Tensor optimized_kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int total_threads = (n + ELEMENTS_PER_THREAD - 1) / ELEMENTS_PER_THREAD;
    const int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int shared_mem_size = BLOCK_SIZE * sizeof(float)/WARP_SIZE;

    optimized_kl_div_kernel<<<blocks, BLOCK_SIZE, shared_mem_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_kl_div_cuda_forward, "Optimized KL divergence forward (CUDA)");
}