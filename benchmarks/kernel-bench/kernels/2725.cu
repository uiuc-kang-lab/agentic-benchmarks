#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Device function for single element LeakyReLU
__device__ __forceinline__ float leaky_relu_device(float x, float negative_slope) {
    return x > 0 ? x : x * negative_slope;
}

// Vectorized kernel processing 4 elements per thread where possible
__global__ void leaky_relu_kernel_hybrid(const float* input, float* output, float negative_slope, 
                                       int n, int n4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process blocks of 4 elements using float4
    if (tid < n4) {
        float4* in4 = (float4*)input;
        float4* out4 = (float4*)output;
        float4 data = in4[tid];
        
        float4 result;
        result.x = leaky_relu_device(data.x, negative_slope);
        result.y = leaky_relu_device(data.y, negative_slope);
        result.z = leaky_relu_device(data.z, negative_slope);
        result.w = leaky_relu_device(data.w, negative_slope);
        
        out4[tid] = result;
    }
    
    // Handle remaining elements in the same kernel
    int remaining_start = n4 * 4;
    int remaining_idx = remaining_start + tid;
    
    if (remaining_idx < n) {
        output[remaining_idx] = leaky_relu_device(input[remaining_idx], negative_slope);
    }
}

torch::Tensor leaky_relu_forward_hybrid(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    int n4 = n / 4;  // Number of float4 elements
    
    const int threads = 256;
    const int blocks = std::max((n + threads - 1) / threads, (n4 + threads - 1) / threads);
    
    leaky_relu_kernel_hybrid<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        negative_slope,
        n,
        n4
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward_hybrid, "LeakyReLU forward hybrid (CUDA)");
}