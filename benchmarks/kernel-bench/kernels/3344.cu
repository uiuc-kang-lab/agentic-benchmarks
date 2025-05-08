#include <torch/extension.h>

#define NUM_STREAMS 4

__global__ void swish_kernel(const float* x, float* y, int64_t n) {
    const int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        const float val = x[index];
        const float sigmoid = 1.0f / (1.0f + expf(-val));
        y[index] = val * sigmoid;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int threads = 256;
    const int blocks_per_stream = ((n + threads - 1) / threads + NUM_STREAMS - 1) / NUM_STREAMS;
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        const int64_t offset = i * blocks_per_stream * threads;
        const int64_t size = min(n - offset, blocks_per_stream * threads);
        if (size > 0) {
            swish_kernel<<<blocks_per_stream, threads, 0, streams[i]>>>(
                x.data_ptr<float>() + offset,
                y.data_ptr<float>() + offset,
                size
            );
        }
    }
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CUDA)");
}