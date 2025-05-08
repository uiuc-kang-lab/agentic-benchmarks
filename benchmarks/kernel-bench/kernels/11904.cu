/*
 * Combined KL Divergence CUDA Kernel with Streaming Support
 * This kernel leverages grid-stride loops with multi-element processing per thread,
 * warp-level reduction using __shfl_down_sync, and shared memory reduction.
 * On the host side, if input tensors are on the GPU, the kernel is launched directly;
 * if the inputs are on the CPU, data is partitioned into chunks and processed via CUDA streams
 * to overlap memory transfers with computation.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Constants
#define WARP_SIZE 32
#define BLOCK_SIZE 256
#define DEFAULT_ELEMENTS_PER_THREAD 4

// For streaming (CPU inputs)
#define CHUNK_SIZE (1 << 16) // 65536 elements per chunk
#define STREAM_COUNT 4

// Device function for warp-level reduction
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Combined kernel: processes multiple elements per thread using a grid-stride loop
// and then reduces the per-thread results using warp and shared memory reductions.
__global__ void kl_div_kernel_combined(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int n,
    int elements_per_thread) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float thread_sum = 0.0f;

    // Each thread processes elements_per_thread consecutive elements per iteration
    // using a grid-stride loop.
    for (int i = tid * elements_per_thread; i < n; i += stride * elements_per_thread) {
        #pragma unroll
        for (int j = 0; j < elements_per_thread; j++) {
            int idx = i + j;
            if (idx < n) {
                float log_val = log_predictions[idx];
                float target = targets[idx];
                thread_sum += expf(log_val) - target * log_val;
            }
        }
    }

    // Intra-warp reduction using shuffle operations
    thread_sum = warp_reduce(thread_sum);

    // Allocate shared memory for storing warp results
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction performed by the first warp of the block
    if (warp_id == 0) {
        float block_sum = (lane < (BLOCK_SIZE / WARP_SIZE)) ? warp_sums[lane] : 0.0f;
        block_sum = warp_reduce(block_sum);
        if (lane == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host forward function that selects between direct GPU kernel launch and streaming
// based on whether the inputs are on the GPU or CPU.

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    // Allocate output on GPU
    auto output = torch::zeros({1}, log_predictions.options().device(torch::kCUDA));

    const int elements_per_thread = DEFAULT_ELEMENTS_PER_THREAD;

    // If inputs are on GPU, launch kernel directly on the full data
    if (log_predictions.is_cuda() && targets.is_cuda()) {
        // Calculate total threads required based on multi-element processing
        int total_threads = (n + elements_per_thread - 1) / elements_per_thread;
        int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int shared_mem_size = (BLOCK_SIZE / WARP_SIZE) * sizeof(float);

        kl_div_kernel_combined<<<blocks, BLOCK_SIZE, shared_mem_size>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n,
            elements_per_thread
        );
        cudaDeviceSynchronize();
        return output / static_cast<float>(n);
    }

    // Otherwise, assume inputs are on CPU and use streaming to overlap transfers with computation
    float* h_log_predictions = log_predictions.data_ptr<float>();
    float* h_targets = targets.data_ptr<float>();

    cudaStream_t streams[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamCreate(&streams[i]);
    }

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

        int total_threads = (current_chunk + elements_per_thread - 1) / elements_per_thread;
        int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int shared_mem_size = (BLOCK_SIZE / WARP_SIZE) * sizeof(float);

        kl_div_kernel_combined<<<blocks, BLOCK_SIZE, shared_mem_size, stream>>>(
            d_log_chunk,
            d_target_chunk,
            output.data_ptr<float>(),
            current_chunk,
            elements_per_thread
        );

        cudaFreeAsync(d_log_chunk, stream);
        cudaFreeAsync(d_target_chunk, stream);

        offset += current_chunk;
    }

    // Synchronize and destroy streams
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA combined)");
}
