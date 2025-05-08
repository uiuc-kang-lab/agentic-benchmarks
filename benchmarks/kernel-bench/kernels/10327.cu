#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Precompute constants to avoid repeated calculations
struct GELUConstants {
    static constexpr float sqrt_2_over_pi = 0.7978845608f;
    static constexpr float coeff = 0.044715f;
};

// Preprocess input value
__device__ __forceinline__ float preprocess_gelu(float x) {
    float x_squared = x * x;
    float x_cubed = x_squared * x;
    return x_cubed * GELUConstants::coeff + x;
}

// Compute inner part of GELU
__device__ __forceinline__ float compute_inner_gelu(float preprocessed_val) {
    return tanhf(preprocessed_val * GELUConstants::sqrt_2_over_pi);
}

// Compute final GELU result
__device__ __forceinline__ float finalize_gelu(float x, float tanh_val) {
    return 0.5f * x * (1.0f + tanh_val);
}

__global__ void gelu_kernel_modular(const float* __restrict__ x, 
                                  float* __restrict__ y,
                                  const int n) {
    extern __shared__ float shared_x[];
    const int tid = threadIdx.x;
    const int unroll = 4;
    const int stride = blockDim.x * unroll;
    const int gid = blockIdx.x * stride + tid;
    
    // Cooperative loading using shared memory
    #pragma unroll
    for (int i = 0; i < unroll; i++) {
        const int idx = gid + i * blockDim.x;
        if (idx < n) {
            shared_x[tid + i * blockDim.x] = x[idx];
        }
    }
    __syncthreads();
    
    // Process elements using modular computation
    #pragma unroll
    for (int i = 0; i < unroll; i++) {
        const int idx = gid + i * blockDim.x;
        if (idx < n) {
            const float xi = shared_x[tid + i * blockDim.x];
            const float preprocessed = preprocess_gelu(xi);
            const float inner_result = compute_inner_gelu(preprocessed);
            y[idx] = finalize_gelu(xi, inner_result);
        }
    }
}

torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    const int n = x.numel();
    
    const int threads = 256;
    const int unroll = 4;
    const int blocks = (n + threads * unroll - 1) / (threads * unroll);
    const size_t shared_mem_size = threads * unroll * sizeof(float);
    
    gelu_kernel_modular<<<blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        n
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "Modular GELU forward CUDA implementation");
}