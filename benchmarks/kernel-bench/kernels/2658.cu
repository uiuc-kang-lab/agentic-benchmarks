#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_kernel_shared(const float* x, float* out, float negative_slope, int n) {
    extern __shared__ float shared_x[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory
    if (idx < n) {
        shared_x[tid] = x[idx];
    } else {
        shared_x[tid] = 0.0f; // Padding for out of bounds
    }
    __syncthreads();

    // Apply LeakyReLU
    if (idx < n) {
        float val = shared_x[tid];
        out[idx] = val > 0 ? val : val * negative_slope;
    }
}

torch::Tensor leaky_relu_forward_shared(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    size_t shared_memory_size = threads * sizeof(float);

    leaky_relu_kernel_shared<<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_shared, "LeakyReLU forward with shared memory (CUDA)");
}