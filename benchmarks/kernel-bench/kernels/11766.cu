#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 8;  // Process more elements per thread

__global__ void strided_warp_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int elements_per_stride = total_threads * ELEMENTS_PER_THREAD;
    const int num_strides = (n + elements_per_stride - 1) / elements_per_stride;
    
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    
    float sum = 0.0f;
    
    // Each thread processes multiple elements with stride pattern
    for (int stride_idx = 0; stride_idx < num_strides; stride_idx++) {
        const int base_idx = stride_idx * elements_per_stride + tid;
        
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; i++) {
            const int idx = base_idx + i * total_threads;
            if (idx < n) {
                const float log_pred = __ldg(log_predictions + idx);
                const float target = __ldg(targets + idx);
                sum += expf(log_pred) - target * log_pred;
            }
        }
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    
    // Store warp results
    extern __shared__ float warp_results[];
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    
    __syncthreads();
    
    // Final reduction by first warp
    if (warp_id == 0) {
        sum = (lane_id < (blockDim.x / WARP_SIZE)) ? warp_results[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor strided_warp_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Optimize thread/block count based on workload
    const int threads = 256;
    const int min_elements_per_block = threads * ELEMENTS_PER_THREAD;
    const int desired_blocks = (n + min_elements_per_block - 1) / min_elements_per_block;
    const int max_blocks = 256;  // Limit max blocks to ensure enough work per block
    const int blocks = min(desired_blocks, max_blocks);
    
    const int shared_mem = (threads / WARP_SIZE) * sizeof(float);
    
    strided_warp_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &strided_warp_kl_forward, "Strided warp-based KL divergence (CUDA)");
}