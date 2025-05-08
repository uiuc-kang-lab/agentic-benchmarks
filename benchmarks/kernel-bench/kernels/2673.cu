#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Combine the benefits of optimized block size and thread count
#define OPTIMAL_THREADS 512

__global__ void leaky_relu_kernel_combined(const float* x, float* out, float negative_slope, int n) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory if within bounds
    if (idx < n) {
        sdata[tid] = x[idx];
    } else {
        sdata[tid] = 0.0f; // Handle out-of-bound threads
    }
    __syncthreads();

    // Apply the LeakyReLU function
    if (idx < n) {
        float val = sdata[tid];
        out[idx] = (val > 0.0f) ? val : val * negative_slope;
    }
}

// Unified forward function
torch::Tensor leaky_relu_forward_combined(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = OPTIMAL_THREADS;  // Use optimal threads
    const int blocks = (n + threads - 1) / threads;
    size_t shared_memory_size = threads * sizeof(float);

    leaky_relu_kernel_combined<<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_combined", &leaky_relu_forward_combined, "LeakyReLU forward combined (CUDA)");
}
