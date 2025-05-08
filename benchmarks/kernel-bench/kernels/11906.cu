#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 4
#define CHUNK_SIZE (1 << 16)  // 65536 elements per chunk
#define STREAM_COUNT 4
#define MIN_ELEMENTS_FOR_STREAMING (1 << 22)  // 4M elements threshold

__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void kl_div_kernel_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n,
    const int elements_per_thread) {
    
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int global_thread_id = blockIdx.x * blockDim.x + tid;
    
    extern __shared__ float warp_results[];
    
    float thread_sum = 0.0f;
    
    // Each thread processes multiple elements with coalesced memory access
    const int start_idx = global_thread_id * elements_per_thread;
    #pragma unroll
    for (int i = 0; i < elements_per_thread; i++) {
        const int idx = start_idx + i;
        if (idx < n) {
            const float log_pred = log_predictions[idx];
            const float target = targets[idx];
            thread_sum += __expf(log_pred) - target * log_pred;  // Using fast math
        }
    }
    
    // Two-level reduction: first within warps, then across warps
    thread_sum = warp_reduce(thread_sum);
    
    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }
    __syncthreads();
    
    if (wid == 0) {
        float warp_sum = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_results[lane] : 0.0f;
        warp_sum = warp_reduce(warp_sum);
        
        if (lane == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options().device(torch::kCUDA));

    if (!log_predictions.is_cuda() && n >= MIN_ELEMENTS_FOR_STREAMING) {
        cudaStream_t streams[STREAM_COUNT];
        for (int i = 0; i < STREAM_COUNT; i++) {
            cudaStreamCreate(&streams[i]);
        }

        float* h_log_predictions = log_predictions.data_ptr<float>();
        float* h_targets = targets.data_ptr<float>();
        
        int offset = 0;
        while (offset < n) {
            int current_chunk = std::min(CHUNK_SIZE, n - offset);
            int stream_idx = (offset / CHUNK_SIZE) % STREAM_COUNT;
            cudaStream_t stream = streams[stream_idx];

            float* d_log_chunk = nullptr;
            float* d_target_chunk = nullptr;
            cudaMallocAsync((void**)&d_log_chunk, current_chunk * sizeof(float), stream);
            cudaMallocAsync((void**)&d_target_chunk, current_chunk * sizeof(float), stream);

            cudaMemcpyAsync(d_log_chunk, h_log_predictions + offset,
                          current_chunk * sizeof(float), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_target_chunk, h_targets + offset,
                          current_chunk * sizeof(float), cudaMemcpyHostToDevice, stream);

            const int elements_per_thread = ELEMENTS_PER_THREAD;
            const int total_threads_needed = (current_chunk + elements_per_thread - 1) / elements_per_thread;
            const int blocks = (total_threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
            const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
            const int shared_mem = warps_per_block * sizeof(float);

            kl_div_kernel_optimized<<<blocks, BLOCK_SIZE, shared_mem, stream>>>(
                d_log_chunk, d_target_chunk, output.data_ptr<float>(),
                current_chunk, elements_per_thread);

            cudaFreeAsync(d_log_chunk, stream);
            cudaFreeAsync(d_target_chunk, stream);
            offset += current_chunk;
        }

        for (int i = 0; i < STREAM_COUNT; i++) {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }
    } else {
        const int elements_per_thread = ELEMENTS_PER_THREAD;
        const int total_threads_needed = (n + elements_per_thread - 1) / elements_per_thread;
        const int blocks = (total_threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
        const int shared_mem = warps_per_block * sizeof(float);

        kl_div_kernel_optimized<<<blocks, BLOCK_SIZE, shared_mem>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n,
            elements_per_thread
        );
    }
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Adaptive KL divergence forward (CUDA)");
}