#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Device function for GELU activation computation
__device__ float compute_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = x + coeff * x_cubed;
    inner *= sqrt_2_over_pi;
    float tanh_val = tanhf(inner);
    return 0.5f * x * (1.0f + tanh_val);
}

// Combined kernel that applies the GELU activation using shared memory and manual unrolling
__global__ void gelu_kernel_combined(const float* __restrict__ x, float* __restrict__ y, int n) {
    extern __shared__ float shared_x[];
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    const int base = blockIdx.x * blockSize * 4; // unroll factor of 4

    // Load data into shared memory with manual unroll
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = base + tid + i * blockSize;
        if (idx < n) {
            shared_x[tid + i * blockSize] = x[idx];
        }
    }
    __syncthreads();

    // Process elements using shared memory
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int idx = base + tid + i * blockSize;
        if (idx < n) {
            y[idx] = compute_gelu(shared_x[tid + i * blockSize]);
        }
    }
}

// Torch binding to launch the combined GELU kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    const int unroll_factor = 4;
    int blocks = (n + threads * unroll_factor - 1) / (threads * unroll_factor);
    size_t shared_mem_size = threads * unroll_factor * sizeof(float);
    
    gelu_kernel_combined<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Combined GELU forward CUDA implementation with shared memory and manual unrolling");
}