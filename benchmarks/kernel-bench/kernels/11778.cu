#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 8;

__global__ void optimized_grid_stride_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = gridDim.x * blockDim.x;
    const int elements_per_cycle = total_threads * ELEMENTS_PER_THREAD;
    
    float thread_sum = 0.0f;

    // Grid-stride loop with predefined element count per iteration
    for (int base = global_tid * ELEMENTS_PER_THREAD; 
         base < n; 
         base += elements_per_cycle) {
        
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int idx = base + i * total_threads;
            if (idx < n) {
                const float log_pred = __ldg(log_predictions + idx);
                const float target = __ldg(targets + idx);
                thread_sum += expf(log_pred) - target * log_pred;
            }
        }
    }

    // Warp-level reduction using butterfly pattern
    for (int offset = WARP_SIZE/2; offset >= 1; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Block-level reduction
    extern __shared__ float warp_sums[];
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (threadIdx.x % WARP_SIZE == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction by first warp only
    if (warp_id == 0) {
        thread_sum = (threadIdx.x < blockDim.x / WARP_SIZE) ? warp_sums[threadIdx.x] : 0.0f;
        for (int offset = WARP_SIZE/2; offset >= 1; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        if (threadIdx.x == 0) {
            atomicAdd(output, thread_sum);
        }
    }
}

torch::Tensor optimized_grid_stride_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int elements_per_block = threads * ELEMENTS_PER_THREAD;
    int blocks = (n + elements_per_block - 1) / elements_per_block;
    blocks = min(blocks, 256);
    
    optimized_grid_stride_kernel<<<blocks, threads, (threads/WARP_SIZE)*sizeof(float)>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_grid_stride_forward, "Optimized grid-stride KL divergence (CUDA)");
}
