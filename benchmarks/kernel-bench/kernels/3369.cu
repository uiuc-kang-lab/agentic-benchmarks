#include <torch/extension.h>
#include <vector>

__global__ void swish_kernel(const float* __restrict__ x, float* __restrict__ y, int64_t n) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float val = x[idx];
        const float sigmoid = 1.0f / (1.0f + expf(-val));
        y[idx] = val * sigmoid;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    const int num_streams = 4;
    const int64_t chunk_size = (n + num_streams - 1) / num_streams;
    
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    const int threads = 256;
    for (int i = 0; i < num_streams; i++) {
        const int64_t offset = i * chunk_size;
        const int64_t current_size = std::min(chunk_size, n - offset);
        if (current_size <= 0) break;
        
        const int blocks = (current_size + threads - 1) / threads;
        swish_kernel<<<blocks, threads, 0, streams[i]>>>(
            x.data_ptr<float>() + offset,
            y.data_ptr<float>() + offset,
            current_size
        );
    }
    
    // Synchronize and cleanup streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation forward pass (CUDA) with streams");
}