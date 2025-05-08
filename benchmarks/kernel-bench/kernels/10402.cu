#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Modular device function to compute GELU activation for a single float
__device__ __forceinline__ float gelu_activate(float x) {
    const float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
    const float coeff = 0.044715f;
    float x_cubed = x * x * x;
    float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
    return 0.5f * x * (1.0f + tanhf(inner));
}

// Modular device function to compute GELU activation on a float4 vector
__device__ __forceinline__ float4 gelu_activate_vec(float4 v) {
    float4 out;
    // Interpret the float4 as an array of 4 floats
    float* vin = reinterpret_cast<float*>(&v);
    float* vout = reinterpret_cast<float*>(&out);
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        vout[i] = gelu_activate(vin[i]);
    }
    return out;
}

// CUDA kernel utilizing vectorized loads/stores and modular device functions
__global__ void gelu_kernel(const float* x, float* y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process elements in chunks of 4 using float4 vectorization
    int vecCount = n / 4;  
    for (int i = tid; i < vecCount; i += stride) {
        float4 in = ((const float4*)x)[i];
        float4 out = gelu_activate_vec(in);
        ((float4*)y)[i] = out;
    }
    
    // Process any remaining elements
    int remainder = vecCount * 4;
    for (int i = remainder + tid; i < n; i += stride) {
        y[i] = gelu_activate(x[i]);
    }
}

// Host function that sets up and launches the kernel
torch::Tensor gelu_forward(torch::Tensor x) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be on CUDA");
    TORCH_CHECK(x.is_contiguous(), "Input tensor must be contiguous");
    
    auto y = torch::empty_like(x);
    int n = x.numel();
    
    // Configure kernel launch parameters
    const int threads = 256;
    const int max_blocks = 1024;
    int blocks = min((n + threads - 1) / threads, max_blocks);
    
    gelu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), y.data_ptr<float>(), n);
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &gelu_forward, "GELU forward CUDA implementation");
}
