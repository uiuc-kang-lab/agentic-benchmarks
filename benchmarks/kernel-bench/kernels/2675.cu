#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function to compute LeakyReLU
__device__ inline float leaky_relu_func(float x, float negative_slope) {
    return (x > 0.0f) ? x : (x * negative_slope);
}

// Device function to load an element from global to shared memory
__device__ inline void load_shared(const float *input, float *shared_data, int idx, int tid, int n) {
    shared_data[tid] = (idx < n) ? input[idx] : 0.0f;
}

// Modular CUDA kernel using shared memory and modular device functions
__global__ void leaky_relu_kernel_modular(const float* x, float* out, float negative_slope, int n) {
    extern __shared__ float shared_x[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load data into shared memory via modular device function
    load_shared(x, shared_x, idx, tid, n);
    __syncthreads();

    // Apply LeakyReLU using the modular device function
    if (idx < n) {
        out[idx] = leaky_relu_func(shared_x[tid], negative_slope);
    }
}

// Forward function that launches the kernel
torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();

    const int threads = 1024;
    const int blocks = (n + threads - 1) / threads;
    size_t shared_memory_size = threads * sizeof(float);

    leaky_relu_kernel_modular<<<blocks, threads, shared_memory_size>>>(
        x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
    );

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "Modular LeakyReLU forward (CUDA)");
}
