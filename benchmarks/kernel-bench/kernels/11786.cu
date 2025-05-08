#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 4;

__global__ void coalesced_chunked_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int element_stride = blockDim.x * gridDim.x * ELEMENTS_PER_THREAD;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    float thread_sum = 0.0f;

    // Process ELEMENTS_PER_THREAD consecutive elements per iteration
    for (int idx_base = tid * ELEMENTS_PER_THREAD; 
         idx_base < n; 
         idx_base += element_stride) {
        
        #pragma unroll
        for (int i = 0; i < ELEMENTS_PER_THREAD; ++i) {
            const int idx = idx_base + i;
            if (idx < n) {
                const float log_pred = __ldg(log_predictions + idx);
                const float target = __ldg(targets + idx);
                thread_sum += expf(log_pred) - target * log_pred;
            }
        }
    }

    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Shared memory buffer for warp sums
    extern __shared__ float warp_sums[];
    if (lane_id == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // First warp reduces all warp contributions
    if (warp_id == 0) {
        float sum = (lane_id < (blockDim.x / WARP_SIZE)) ? warp_sums[lane_id] : 0.0f;
        
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane_id == 0) {
            atomicAdd(output, sum);
        }
    }
}

torch::Tensor coalesced_chunked_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int elements_per_block = threads * ELEMENTS_PER_THREAD;
    const int desired_blocks = (n + elements_per_block - 1) / elements_per_block;
    const int max_blocks = 512;
    const int blocks = min(desired_blocks, max_blocks);
    const int shared_mem = (threads / WARP_SIZE) * sizeof(float);

    coalesced_chunked_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &coalesced_chunked_kl_forward, "Coalesced chunked KL divergence (CUDA)");
}
