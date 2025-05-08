#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        throw std::runtime_error(cudaGetErrorString(err)); \
    } \
} while(0)

// RAII wrapper for CUDA stream
class CudaStreamGuard {
public:
    CudaStreamGuard() { CUDA_CHECK(cudaStreamCreate(&stream_)); }
    ~CudaStreamGuard() { cudaStreamDestroy(stream_); }
    cudaStream_t get() const { return stream_; }
private:
    cudaStream_t stream_;
};

// CUDA kernel for ReLU activation
template <typename scalar_t>
__global__ void relu_kernel(
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ input,
    const int64_t size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < size) {
        const scalar_t val = input[idx];
        output[idx] = val > 0 ? val : 0;
    }
}

// PyTorch wrapper function
torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    const int blocks = (input.numel() + threads - 1) / threads;
    
    CudaStreamGuard stream_guard;
    auto stream = stream_guard.get();

    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_kernel", ([&] {
        relu_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            input.numel()
        );
    }));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "ReLU forward with streams (CUDA)");
}