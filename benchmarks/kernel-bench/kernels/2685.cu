#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm> // for std::min

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel for LeakyReLU operation on a chunk of the tensor
__global__ void leaky_relu_kernel_overlap(const float* x, float* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        out[idx] = (val > 0.0f) ? val : val * negative_slope;
    }
}

// Forward function that splits the tensor into chunks and processes them using multiple CUDA streams
torch::Tensor leaky_relu_forward_overlap(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    // Define the chunk size (number of elements per chunk) for pipelining
    int chunk_size = 1 << 20; // 1M elements per chunk
    int num_chunks = (n + chunk_size - 1) / chunk_size;

    // Use a fixed number of streams (e.g., 4) to overlap computation and memory transfers
    int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;
    
    // Process each chunk asynchronously on a rotating set of streams
    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int offset = chunk * chunk_size;
        int current_chunk = std::min(chunk_size, n - offset);
        int blocks = (current_chunk + threads - 1) / threads;
        int stream_idx = chunk % num_streams;

        leaky_relu_kernel_overlap<<<blocks, threads, 0, streams[stream_idx]>>>(
            x.data_ptr<float>() + offset,
            out.data_ptr<float>() + offset,
            negative_slope,
            current_chunk
        );
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_overlap, "LeakyReLU forward with overlapped computation and memory transfers (CUDA)");
}
