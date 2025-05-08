#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void leaky_relu_optimized_kernel(const float* __restrict__ x, float* __restrict__ out, float negative_slope, int n) {
    extern __shared__ float shared_x[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int tid = threadIdx.x;

    // Load data into shared memory with grid-stride loop
    for (int i = idx; i < n; i += stride) {
        shared_x[tid] = x[i];
        __syncthreads();

        // Apply LeakyReLU
        float val = shared_x[tid];
        out[i] = val > 0 ? val : val * negative_slope;
        __syncthreads();
    }
}

torch::Tensor leaky_relu_optimized_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;

    size_t shared_memory_size = threads * sizeof(float);

    leaky_relu_optimized_kernel<<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_optimized_forward, "LeakyReLU forward with optimized shared memory and grid-stride loop (CUDA)");
}