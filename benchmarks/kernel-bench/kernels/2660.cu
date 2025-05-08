#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel that applies LeakyReLU activation on a chunk of data
__global__ void leaky_relu_kernel_chunk(const float* x, float* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = v > 0 ? v : v * negative_slope;
    }
}

// Pipelined LeakyReLU forward function using asynchronous memory transfers
// This implementation expects the input tensor to reside in CPU pinned memory so that
// the host-to-device and device-to-host copies can be performed asynchronously.
// The input is partitioned into chunks and processed using 2 CUDA streams to overlap
// memory transfers with kernel execution, reducing overall runtime while maintaining full precision.

torch::Tensor pipelined_leaky_relu_forward(torch::Tensor x, float negative_slope) {
    // Ensure the input tensor is on CPU, contiguous, and pinned for asynchronous transfer
    TORCH_CHECK(x.device().is_cpu(), "pipelined_leaky_relu: input must be a CPU tensor");
    TORCH_CHECK(x.is_contiguous(), "pipelined_leaky_relu: input must be contiguous");
    TORCH_CHECK(x.is_pinned(), "pipelined_leaky_relu: input must be pinned memory");

    int n = x.numel();
    
    // Create output tensor in pinned CPU memory to enable asynchronous device-to-host copy
    auto opts = torch::TensorOptions().dtype(x.dtype()).device(torch::kCPU).pinned_memory(true);
    auto out = torch::empty_like(x, opts);

    // Define chunk parameters: process input in chunks to pipeline transfers and computation
    int chunk_size = 1 << 20; // 1M elements per chunk
    int num_chunks = (n + chunk_size - 1) / chunk_size;
    int num_streams = 2; // Using double buffering

    // Create CUDA streams
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate device buffers for each stream (allocate maximum chunk size)
    std::vector<float*> d_buffers(num_streams, nullptr);
    size_t buf_size = chunk_size * sizeof(float);
    for (int i = 0; i < num_streams; i++) {
        cudaMalloc(&d_buffers[i], buf_size);
    }

    // Process each chunk in a pipelined manner
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        int stream_idx = chunk % num_streams;
        int offset = chunk * chunk_size;
        int current_chunk_size = std::min(chunk_size, n - offset);
        size_t bytes = current_chunk_size * sizeof(float);

        // Asynchronously copy the current chunk from host (pinned) to device
        const float* src_ptr = x.data_ptr<float>() + offset;
        cudaMemcpyAsync(d_buffers[stream_idx], src_ptr, bytes, cudaMemcpyHostToDevice, streams[stream_idx]);

        // Launch the kernel on the device for this chunk
        int threads = 1024;
        int blocks = (current_chunk_size + threads - 1) / threads;
        leaky_relu_kernel_chunk<<<blocks, threads, 0, streams[stream_idx]>>>(
            d_buffers[stream_idx], d_buffers[stream_idx], negative_slope, current_chunk_size
        );

        // Asynchronously copy the result from device back to host (pinned output tensor)
        float* dst_ptr = out.data_ptr<float>() + offset;
        cudaMemcpyAsync(dst_ptr, d_buffers[stream_idx], bytes, cudaMemcpyDeviceToHost, streams[stream_idx]);
    }

    // Synchronize all streams to ensure computation and transfers are complete
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    // Free device buffers
    for (int i = 0; i < num_streams; i++) {
        cudaFree(d_buffers[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pipelined_leaky_relu_forward, "Pipelined LeakyReLU forward (CUDA, asynchronous transfers)");
}
