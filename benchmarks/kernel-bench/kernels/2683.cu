#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel to process a chunk of data
__global__ void leaky_relu_kernel_chunk(const float* in, float* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        out[idx] = (val > 0.0f) ? val : val * negative_slope;
    }
}

// Forward function that overlaps computation with memory transfers using CUDA streams with double buffering
torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto y = torch::empty_like(x);
    int n = x.numel();

    // Define chunk size for pipelining. Here we use 1<<20 (approximately 1 million elements) per chunk.
    int chunk_size = 1 << 20;
    if (chunk_size > n) {
        chunk_size = n;
    }
    int num_chunks = (n + chunk_size - 1) / chunk_size;

    // Allocate two device buffers for double buffering
    float* d_buffer[2];
    size_t buffer_bytes = chunk_size * sizeof(float);
    cudaMalloc(&d_buffer[0], buffer_bytes);
    cudaMalloc(&d_buffer[1], buffer_bytes);

    // Create two CUDA streams: one for asynchronous copy and one for kernel execution
    cudaStream_t stream_copy, stream_compute;
    cudaStreamCreate(&stream_copy);
    cudaStreamCreate(&stream_compute);

    // Create events for synchronizing copy and compute streams
    cudaEvent_t events[2];
    cudaEventCreate(&events[0]);
    cudaEventCreate(&events[1]);

    // Kernel launch parameters for each chunk
    int threads = 256;
    int blocks = (chunk_size + threads - 1) / threads;

    // Pipeline: for each chunk, copy data asynchronously and then launch kernel
    for (int i = 0; i < num_chunks; i++) {
        int offset = i * chunk_size;
        int current_chunk = ((offset + chunk_size) <= n) ? chunk_size : (n - offset);
        int buf_idx = i % 2;  // alternate between the two buffers

        // Asynchronously copy the current chunk from input tensor to the double buffer
        cudaMemcpyAsync(d_buffer[buf_idx],
                        x.data_ptr<float>() + offset,
                        current_chunk * sizeof(float),
                        cudaMemcpyDeviceToDevice,
                        stream_copy);

        // Record event on copy stream after the copy finishes
        cudaEventRecord(events[buf_idx], stream_copy);

        // Make compute stream wait until the copy for this buffer is done
        cudaStreamWaitEvent(stream_compute, events[buf_idx], 0);

        // Launch kernel on the copied chunk in the compute stream
        leaky_relu_kernel_chunk<<<blocks, threads, 0, stream_compute>>>(
            d_buffer[buf_idx],
            y.data_ptr<float>() + offset,
            negative_slope,
            current_chunk
        );
    }

    // Synchronize both streams to ensure all operations are complete
    cudaStreamSynchronize(stream_copy);
    cudaStreamSynchronize(stream_compute);

    // Cleanup: destroy events, streams, and free the double buffers
    cudaEventDestroy(events[0]);
    cudaEventDestroy(events[1]);
    cudaStreamDestroy(stream_copy);
    cudaStreamDestroy(stream_compute);
    cudaFree(d_buffer[0]);
    cudaFree(d_buffer[1]);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward with overlapping computation and memory transfers (CUDA)");
}
