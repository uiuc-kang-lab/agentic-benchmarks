#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Optimized KL divergence kernel performing reduction with warp-shuffle operations
__global__ void optimized_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    int n) {

    const unsigned int warp_size = 32;
    const unsigned int lane_id = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    const unsigned int warps_per_block = blockDim.x / warp_size;

    extern __shared__ float warp_results[];

    float sum = 0.0f;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        float log_pred = log_predictions[i];
        float target = targets[i];
        sum += expf(log_pred) - target * log_pred;
    }

    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // First thread in each warp writes its result to shared memory
    if (lane_id == 0) {
        warp_results[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction across warps, done by first warp
    if (warp_id == 0) {
        float warp_sum = (lane_id < warps_per_block) ? warp_results[lane_id] : 0.0f;
        #pragma unroll
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        if (lane_id == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

// Forward function using CUDA streams to overlap memory transfers with kernel execution
// Assumes that input tensors (log_predictions and targets) reside in pinned host memory
// to benefit from asynchronous memory copy.

torch::Tensor kl_div_cuda_forward_streamed(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    // Total number of elements
    const int n = log_predictions.numel();

    // Create device output tensor and initialize to zero
    auto options = log_predictions.options().device(torch::kCUDA);
    auto output = torch::zeros({1}, options);

    // Set up parameters for pipelining
    constexpr int CHUNK_SIZE = 1 << 20; // Process 1M elements per chunk
    int num_chunks = (n + CHUNK_SIZE - 1) / CHUNK_SIZE;
    int num_streams = 4; // Use 4 CUDA streams for overlapping data transfer and computation

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Preallocate device buffers for each stream
    std::vector<float*> d_log(num_streams, nullptr);
    std::vector<float*> d_targ(num_streams, nullptr);
    size_t buffer_bytes = CHUNK_SIZE * sizeof(float);
    for (int i = 0; i < num_streams; i++) {
        cudaMalloc((void**)&d_log[i], buffer_bytes);
        cudaMalloc((void**)&d_targ[i], buffer_bytes);
    }

    // Kernel launch configuration
    const int threads = 256;

    // Process input data in chunks, overlapping host-to-device copies with kernel execution
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int current_chunk_size = std::min(CHUNK_SIZE, n - chunk * CHUNK_SIZE);
        int stream_idx = chunk % num_streams;

        // Pointers to the current chunk in host memory
        const float* h_log = log_predictions.data_ptr<float>() + chunk * CHUNK_SIZE;
        const float* h_targ = targets.data_ptr<float>() + chunk * CHUNK_SIZE;

        // Asynchronously copy the current chunk from host (pinned) to device memory
        cudaMemcpyAsync(d_log[stream_idx], h_log, current_chunk_size * sizeof(float),
                        cudaMemcpyHostToDevice, streams[stream_idx]);
        cudaMemcpyAsync(d_targ[stream_idx], h_targ, current_chunk_size * sizeof(float),
                        cudaMemcpyHostToDevice, streams[stream_idx]);

        // Determine grid configuration for this chunk
        int blocks = (current_chunk_size + threads - 1) / threads;
        int warps_per_block = threads / 32;
        int shared_mem = warps_per_block * sizeof(float);

        // Launch the KL divergence kernel asynchronously on the selected stream
        optimized_kl_div_kernel<<<blocks, threads, shared_mem, streams[stream_idx]>>>(
            d_log[stream_idx], d_targ[stream_idx], output.data_ptr<float>(), current_chunk_size);
    }

    // Synchronize all streams to ensure all work is complete
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
        cudaFree(d_log[i]);
        cudaFree(d_targ[i]);
    }

    // Finalize the result by dividing the accumulated sum by the total number of elements
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_streamed", &kl_div_cuda_forward_streamed, "KL divergence forward (CUDA with streams)");
}
