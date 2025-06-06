#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel(const float* x, float* out, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        out[idx] = x[idx] / (1.0f + fabsf(x[idx]));
    }
}

// Optimized forward function that combines the simplicity of the first kernel with the stream-based parallelism of the second.
torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int total_elements = x.numel();

    // Set chunk size (number of elements per stream). This size can be tuned for best performance.
    int chunk_size = 1 << 20; // Example: 1M elements per chunk
    int num_chunks = (total_elements + chunk_size - 1) / chunk_size;
    int threads = 1024;

    // Create a vector to hold CUDA streams
    std::vector<cudaStream_t> streams(num_chunks);

    // Launch kernel for each chunk on its own stream
    for (int i = 0; i < num_chunks; i++) {
        cudaStreamCreate(&streams[i]);
        int offset = i * chunk_size;
        int current_chunk = std::min(chunk_size, total_elements - offset);
        int blocks = (current_chunk + threads - 1) / threads;
        softsign_kernel<<<blocks, threads, 0, streams[i]>>>(x.data_ptr<float>() + offset, out.data_ptr<float>() + offset, current_chunk);
    }

    // Ensure all streams have completed execution
    for (int i = 0; i < num_chunks; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Softsign activation (CUDA with stream overlap)");
}
