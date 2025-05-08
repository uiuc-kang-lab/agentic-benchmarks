#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define BLOCK_SIZE 256
#define CHUNK_SIZE (1 << 16)  // 65536 elements per chunk
#define STREAM_COUNT 4

// Kernel to compute KL divergence over a chunk of data
__global__ void kl_div_kernel_streamed(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    int stride = blockDim.x * gridDim.x;
    for (; idx < n; idx += stride) {
        float log_val = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_val) - target * log_val;
    }

    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    sdata[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0)
        atomicAdd(output, sdata[0]);
}


// Host function using CUDA streams to overlap memory transfers with computation
// If inputs are already on GPU, the kernel is launched directly.

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    int n = log_predictions.numel();

    // Allocate output tensor on the CUDA device
    auto output = torch::zeros({1}, log_predictions.options().device(torch::kCUDA));

    // If inputs are on GPU, launch kernel directly on full data
    if (log_predictions.is_cuda() && targets.is_cuda()) {
        int threads = BLOCK_SIZE;
        int blocks = (n + threads - 1) / threads;
        kl_div_kernel_streamed<<<blocks, threads>>>(
            log_predictions.data_ptr<float>(),
            targets.data_ptr<float>(),
            output.data_ptr<float>(),
            n);
        cudaDeviceSynchronize();
        return output / static_cast<float>(n);
    }

    // Otherwise, assume inputs are in CPU memory and use streaming to overlap transfers & computation
    float* h_log_predictions = log_predictions.data_ptr<float>();
    float* h_targets = targets.data_ptr<float>();

    // Create CUDA streams
    cudaStream_t streams[STREAM_COUNT];
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int offset = 0;
    while (offset < n) {
        int current_chunk = std::min(CHUNK_SIZE, n - offset);
        int stream_idx = (offset / CHUNK_SIZE) % STREAM_COUNT;
        cudaStream_t stream = streams[stream_idx];

        // Allocate device memory for the current chunk using cudaMallocAsync
        float* d_log_chunk = nullptr;
        float* d_target_chunk = nullptr;
        cudaMallocAsync((void**)&d_log_chunk, current_chunk * sizeof(float), stream);
        cudaMallocAsync((void**)&d_target_chunk, current_chunk * sizeof(float), stream);

        // Asynchronously copy the current chunk from host to device
        cudaMemcpyAsync(d_log_chunk, h_log_predictions + offset,
                        current_chunk * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(d_target_chunk, h_targets + offset,
                        current_chunk * sizeof(float), cudaMemcpyHostToDevice, stream);

        // Launch the kernel for the current chunk
        int threads = BLOCK_SIZE;
        int blocks = (current_chunk + threads - 1) / threads;
        kl_div_kernel_streamed<<<blocks, threads, 0, stream>>>(
            d_log_chunk, d_target_chunk, output.data_ptr<float>(), current_chunk);

        // Free device memory asynchronously
        cudaFreeAsync(d_log_chunk, stream);
        cudaFreeAsync(d_target_chunk, stream);

        offset += current_chunk;
    }

    // Synchronize all streams
    for (int i = 0; i < STREAM_COUNT; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with streaming (CUDA)");
}
