#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel_coalesced(const float* __restrict__ x, float* __restrict__ out, float alpha, int n) {
    // Process 4 elements per thread for better memory coalescing
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    
    // Handle 4 elements at a time using float4
    for (int i = tid * 4; i < n - 3; i += stride * 4) {
        float4 in_val = *reinterpret_cast<const float4*>(x + i);
        float4 out_val;
        
        // Process each component
        out_val.x = (in_val.x > 0) ? in_val.x : alpha * (expf(in_val.x) - 1);
        out_val.y = (in_val.y > 0) ? in_val.y : alpha * (expf(in_val.y) - 1);
        out_val.z = (in_val.z > 0) ? in_val.z : alpha * (expf(in_val.z) - 1);
        out_val.w = (in_val.w > 0) ? in_val.w : alpha * (expf(in_val.w) - 1);
        
        // Write result back to global memory
        *reinterpret_cast<float4*>(out + i) = out_val;
    }
    
    // Handle remaining elements
    for (int i = tid * 4 + ((n / 4) * 4); i < n; i += stride) {
        out[i] = (x[i] > 0) ? x[i] : alpha * (expf(x[i]) - 1);
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    // Adjust block size to ensure good occupancy and alignment
    const int threads = 128;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);
    
    elu_kernel_coalesced<<<blocks, threads>>>(x.data_ptr<float>(), out.data_ptr<float>(), alpha, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda, "ELU activation (CUDA)");
}