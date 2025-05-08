#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for ReLU calculation
__device__ float relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

// Kernel function performing ReLU with stream optimization
__global__ void pipelined_relu_kernel(float* __restrict__ output, const float* __restrict__ input, size_t n) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = index; i < n; i += stride) {
        output[i] = relu(input[i]);
    }
}

// PyTorch binding function using CUDA streams
void relu_with_stream(cudaStream_t stream, torch::Tensor output, torch::Tensor input) {
    const auto n = input.numel();
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "pipelined_relu_kernel", ([&] {
        pipelined_relu_kernel<<<blocks, threads, 0, stream>>>(
            output.data_ptr<float>(),
            input.data_ptr<float>(),
            n
        );
    }));
}

// PyTorch wrapper function with stream management
torch::Tensor forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

    relu_with_stream(stream, output, input);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stream-optimized ReLU (CUDA)");
}