#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Threshold for deciding when to use shared memory
#define SHARED_MEMORY_THRESHOLD 1048576  // 1M elements
#define BLOCK_SIZE 256  // Compromise between both versions

template<bool UseShared>
__global__ void leaky_relu_kernel_adaptive(const float* x, float* out, float negative_slope, int n) {
    if constexpr (UseShared) {
        extern __shared__ float sdata[];
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int tid = threadIdx.x;

        if (idx < n) {
            sdata[tid] = x[idx];
        }
        __syncthreads();

        if (idx < n) {
            float val = sdata[tid];
            out[idx] = (val > 0.0f) ? val : val * negative_slope;
        }
    } else {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            float val = x[idx];
            out[idx] = (val > 0.0f) ? val : val * negative_slope;
        }
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = BLOCK_SIZE;
    const int blocks = (n + threads - 1) / threads;
    
    // Choose implementation based on input size
    if (n >= SHARED_MEMORY_THRESHOLD) {
        size_t shared_memory_size = threads * sizeof(float);
        leaky_relu_kernel_adaptive<true><<<blocks, threads, shared_memory_size>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
        );
    } else {
        leaky_relu_kernel_adaptive<false><<<blocks, threads, 0>>>(
            x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, n
        );
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "Adaptive LeakyReLU forward (CUDA)");
}