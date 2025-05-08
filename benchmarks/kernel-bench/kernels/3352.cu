#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// This kernel processes a chunk of data applying the Swish activation
__global__ void swish_chunk_kernel(const float* in, float* out, int64_t current_chunk_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < current_chunk_size) {
        float val = in[idx];
        float sig = 1.0f / (1.0f + expf(-val));
        out[idx] = val * sig;
    }
}

// Host function that splits the input tensor into chunks and overlaps memory copies
// with kernel execution using two CUDA streams.
torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    int64_t n = x.numel();
    auto y = torch::empty_like(x);

    // Choose a chunk size; tune based on problem size, here 2^20 elements (approx 1M) per chunk
    int64_t chunk_size = 1 << 20;
    int64_t num_chunks = (n + chunk_size - 1) / chunk_size;

    // Create two CUDA streams to enable pipelining
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // Allocate temporary device buffers for double buffering
    float *d_buffer_in, *d_buffer_out;
    cudaMalloc(&d_buffer_in, chunk_size * sizeof(float));
    cudaMalloc(&d_buffer_out, chunk_size * sizeof(float));

    // Process each chunk
    for (int64_t chunk = 0; chunk < num_chunks; ++chunk) {
        int64_t offset = chunk * chunk_size;
        int64_t current_chunk_size = ((offset + chunk_size) > n) ? (n - offset) : chunk_size;

        // Alternate streams for overlapping operations
        cudaStream_t stream = (chunk & 1) ? stream1 : stream0;

        // Asynchronously copy the current chunk from input tensor (device memory) into temporary buffer
        cudaMemcpyAsync(d_buffer_in, x.data_ptr<float>() + offset, current_chunk_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);

        // Launch the swish kernel on the current chunk
        int threads = 256;
        int blocks = (current_chunk_size + threads - 1) / threads;
        swish_chunk_kernel<<<blocks, threads, 0, stream>>>(d_buffer_in, d_buffer_out, current_chunk_size);

        // Asynchronously copy the computed results from the temporary output buffer to the output tensor
        cudaMemcpyAsync(y.data_ptr<float>() + offset, d_buffer_out, current_chunk_size * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    // Synchronize both streams to ensure all operations complete before returning
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    // Clean up temporary resources
    cudaFree(d_buffer_in);
    cudaFree(d_buffer_out);
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass with overlapping memory transfers (CUDA)");
}
