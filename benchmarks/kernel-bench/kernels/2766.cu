#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that processes data in groups of 4 using vectorized loads/stores
__global__ void vectorized_leaky_relu_kernel(const float4* __restrict__ x, float4* __restrict__ out, float negative_slope, int vCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < vCount) {
        float4 val = x[idx];
        float4 res;
                res.x = (val.x > 0.0f ? val.x : val.x * negative_slope);
        res.y = (val.y > 0.0f ? val.y : val.y * negative_slope);
        res.z = (val.z > 0.0f ? val.z : val.z * negative_slope);
        res.w = (val.w > 0.0f ? val.w : val.w * negative_slope);
        out[idx] = res;
    }
}

// Kernel to process remaining elements that are not a multiple of 4
__global__ void remainder_leaky_relu_kernel(const float* __restrict__ x, float* __restrict__ out, float negative_slope, int offset, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int global_idx = offset + idx;
        float val = x[global_idx];
        out[global_idx] = (val > 0.0f ? val : val * negative_slope);
    }
}

torch::Tensor leaky_relu_forward(torch::Tensor x, float negative_slope) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    // Calculate how many full vectorized (float4) elements we can process
    int vCount = n / 4;
    int remainder = n % 4;
    
    // Launch vectorized kernel if there is at least one full float4 element
    if (vCount > 0) {
        const int threads = 1024;
        int blocks = (vCount + threads - 1) / threads;
        const float4* x_vec = reinterpret_cast<const float4*>(x.data_ptr<float>());
        float4* out_vec = reinterpret_cast<float4*>(out.data_ptr<float>());
        vectorized_leaky_relu_kernel<<<blocks, threads>>>(x_vec, out_vec, negative_slope, vCount);
    }
    
    // Process any remaining elements
    if (remainder > 0) {
        const int threads = 256;
        int blocks = (remainder + threads - 1) / threads;
        int offset = vCount * 4;
        remainder_leaky_relu_kernel<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), negative_slope, offset, remainder);
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &leaky_relu_forward, "LeakyReLU forward (CUDA) with vectorized global memory accesses");
}
