#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel_vec4(const float4* x, float4* out, float alpha, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n4) {
        float4 val = x[idx];
        float4 result;
        
        result.x = (val.x > 0) ? val.x : alpha * (expf(val.x) - 1);
        result.y = (val.y > 0) ? val.y : alpha * (expf(val.y) - 1);
        result.z = (val.z > 0) ? val.z : alpha * (expf(val.z) - 1);
        result.w = (val.w > 0) ? val.w : alpha * (expf(val.w) - 1);
        
        out[idx] = result;
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    
    int n = x.numel();
    int n4 = n / 4;  // Number of float4 elements
    
    const int threads = 256;
    const int blocks = (n4 + threads - 1) / threads;
    
    // Handle the main part of the array with float4
    elu_kernel_vec4<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        alpha,
        n4
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda, "ELU activation (CUDA)");
}