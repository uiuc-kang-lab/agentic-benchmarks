#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void swish_stream_kernel(const float* x, float* y, int64_t start, int64_t end) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = start + idx; i < end; i += stride) {
        float val = x[i];
        float sig = 1.0f / (1.0f + expf(-val));
        y[i] = val * sig;
    }
}

torch::Tensor swish_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    auto y = torch::empty_like(x);
    const int64_t n = x.numel();
    
    const int num_streams = 4;
    cudaStream_t streams[num_streams];
    const int threads = 256;
    
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    int64_t chunk_size = (n + num_streams - 1) / num_streams;
    
    for (int i = 0; i < num_streams; ++i) {
        int64_t start = i * chunk_size;
        int64_t end = fmin(start + chunk_size, n);
        if (start >= end) break;
        
        int blocks = (end - start + threads - 1) / threads;
        swish_stream_kernel<<<blocks, threads, 0, streams[i]>>>(
            x.data_ptr<float>(),
            y.data_ptr<float>(),
            start,
            end
        );
    }

    for (int i = 0; i < num_streams; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish activation with stream optimization");
}