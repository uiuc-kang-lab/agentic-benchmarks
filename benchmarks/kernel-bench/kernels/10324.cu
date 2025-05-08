#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Inline device function for GELU activation computation
__device__ __forceinline__ float compute_gelu(float x) {
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = (x + coeff * x_cubed) * sqrt_2_over_pi;
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Optimized kernel with uniform control flow to minimize warp divergence
__global__ void gelu_kernel_uniform(const float* __restrict__ x, float* __restrict__ y, int n) {
    extern __shared__ float shared_x[];
    
    const int unroll = 4;
    int tid = threadIdx.x;
    int base = blockIdx.x * blockDim.x * unroll;
    
    // Check if the current block has a full tile of valid elements
    bool full_tile = (base + blockDim.x * unroll <= n);

    if (full_tile) {
        // All accesses are valid; no branch divergence inside the loop
        #pragma unroll
        for (int i = 0; i < unroll; i++) {
            int idx = base + tid + i * blockDim.x;
            shared_x[tid + i * blockDim.x] = x[idx];
        }
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < unroll; i++) {
            int idx = base + tid + i * blockDim.x;
            float xi = shared_x[tid + i * blockDim.x];
            y[idx] = compute_gelu(xi);
        }
    } else {
        // For the tail block, use conditional code to guard against out-of-bound accesses
        #pragma unroll
        for (int i = 0; i < unroll; i++) {
            int idx = base + tid + i * blockDim.x;
            if (idx < n) {
                shared_x[tid + i * blockDim.x] = x[idx];
            }
        }
        __syncthreads();
        
        #pragma unroll
        for (int i = 0; i < unroll; i++) {
            int idx = base + tid + i * blockDim.x;
            if (idx < n) {
                float xi = shared_x[tid + i * blockDim.x];
                y[idx] = compute_gelu(xi);
            }
        }
    }
}

// Host function to launch the kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    const int unroll = 4;
    int blocks = (n + threads * unroll - 1) / (threads * unroll);
    size_t shared_mem_size = threads * unroll * sizeof(float);
    
    gelu_kernel_uniform<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA kernel with uniform control flow to minimize warp divergence");
}
