#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define NUM_STREAMS 4

__global__ void leaky_relu_kernel(const float* x, float* out, float negative_slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] > 0 ? x[idx] : x[idx] * negative_slope;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();
    int chunk_size = (n + NUM_STREAMS - 1) / NUM_STREAMS;

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    const int threads = 256;
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        int offset = i * chunk_size;
        int current_size = min(chunk_size, n - offset);
        if (current_size <= 0) break;
        
        const int blocks = (current_size + threads - 1) / threads;
        
        leaky_relu_kernel<<<blocks, threads, 0, streams[i]>>>(
            x.data_ptr<float>() + offset,
            out.data_ptr<float>() + offset,
            negative_slope,
            current_size
        );
    }

    // Synchronize all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA)");
}