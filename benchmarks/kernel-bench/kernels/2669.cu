#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Declare constant memory for frequently accessed parameter
__constant__ float d_negative_slope;

__global__ void leaky_relu_kernel_optimized(const float* x, float* out, int n) {
    extern __shared__ float shared_x[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Collaborative loading into shared memory
    if (idx < n) {
        shared_x[tid] = x[idx];
    }
    __syncthreads();

    // Process elements using constant memory for negative_slope
    if (idx < n) {
        float val = shared_x[tid];
        // Use constant memory d_negative_slope instead of parameter
        out[idx] = val > 0 ? val : val * d_negative_slope;
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();

    // Copy negative_slope to constant memory
    cudaMemcpyToSymbol(d_negative_slope, &negative_slope, sizeof(float));

    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;
    size_t shared_mem_size = threads * sizeof(float);

    leaky_relu_kernel_optimized<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward with constant memory (CUDA)");
}