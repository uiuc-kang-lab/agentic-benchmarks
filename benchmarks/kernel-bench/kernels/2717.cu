#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Process 4 elements per thread using float4
__global__ void leaky_relu_kernel_vectorized(const float4* x, float4* out, float negative_slope, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 in4 = x[idx];
        float4 result;
        
        result.x = in4.x > 0 ? in4.x : in4.x * negative_slope;
        result.y = in4.y > 0 ? in4.y : in4.y * negative_slope;
        result.z = in4.z > 0 ? in4.z : in4.z * negative_slope;
        result.w = in4.w > 0 ? in4.w : in4.w * negative_slope;
        
        out[idx] = result;
    }
}

torch::Tensor leaky_relu_forward_vectorized(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    int n4 = n / 4;  // Number of float4 elements
    
    const int threads = 256;
    const int blocks = (n4 + threads - 1) / threads;
    
    leaky_relu_kernel_vectorized<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        negative_slope,
        n4
    );
    
    // Handle remaining elements if n is not divisible by 4
    if (n % 4 != 0) {
        int remainder_start = n4 * 4;
        for (int i = remainder_start; i < n; i++) {
            float val = x.data_ptr<float>()[i];
            out.data_ptr<float>()[i] = val > 0 ? val : val * negative_slope;
        }
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_vectorized, "LeakyReLU forward vectorized (CUDA)");
}