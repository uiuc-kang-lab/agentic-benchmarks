#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_kernel_optimized(const float* x, float* out, float slope, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] > 0 ? x[idx] : x[idx] * slope;
    }
}

torch::Tensor leaky_relu_forward_optimized(torch::Tensor x, float slope) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 256;
    const int chunk_size = 1 << 20;
    const int num_streams = 4;
    std::vector<cudaStream_t> streams(num_streams);
    
    for (int i = 0; i < num_streams; ++i)
        cudaStreamCreate(&streams[i]);

    for (int chunk = 0; chunk < (n + chunk_size - 1)/chunk_size; ++chunk) {
        int offset = chunk * chunk_size;
        int curr_size = std::min(chunk_size, n - offset);
        int blocks = (curr_size + threads - 1) / threads;
        
        leaky_relu_kernel_optimized<<<blocks, threads, 0, streams[chunk % num_streams]>>>(
            x.data_ptr<float>() + offset,
            out.data_ptr<float>() + offset,
            slope,
            curr_size
        );
    }

    for (auto& stream : streams) {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_optimized, "Optimized LeakyReLU with stream overlap");
}