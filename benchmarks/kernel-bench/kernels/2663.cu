#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Define the block size based on experimentation
#define BLOCK_SIZE 512

// Kernel using shared memory with optimized block size
__global__ void leaky_relu_kernel_opt(const float* x, float* out, float negative_slope, int n) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
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

// Forward function that launches the optimized kernel
torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;
    size_t shared_memory_size = threads * sizeof(float);

    leaky_relu_kernel_opt<<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward optimized with block size 512 (CUDA)");
}
