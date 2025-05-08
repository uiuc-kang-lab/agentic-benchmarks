#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_kernel_optimized(const float* x, float* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        out[idx] = fmaxf(val, val * negative_slope);
    }
}

torch::Tensor leaky_relu_forward_optimized(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024;
    int chunk_size = (1 << 20) / threads * threads; // Align chunk size with block boundaries
    int num_chunks = (n + chunk_size - 1) / chunk_size;

    int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
        int offset = chunk * chunk_size;
        int current_chunk = std::min(chunk_size, n - offset);
        int blocks = (current_chunk + threads - 1) / threads;
        int stream_idx = chunk % num_streams;

        leaky_relu_kernel_optimized<<<blocks, threads, 0, streams[stream_idx]>>>(
            x.data_ptr<float>() + offset,
            out.data_ptr<float>() + offset,
            negative_slope,
            current_chunk
        );
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_optimized, "Optimized LeakyReLU with streamed 1024-thread blocks");
}