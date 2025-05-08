#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel_coalesced(const float4* __restrict__ x, 
                                   float4* __restrict__ out,
                                   float alpha,
                                   int n4) {
    const int tid = threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Grid-stride loop to ensure coalesced access within warps
    while (idx < n4) {
        float4 val = x[idx];
        float4 result;
        
        // Process elements
        result.x = (val.x > 0.0f) ? val.x : alpha * (expf(val.x) - 1.0f);
        result.y = (val.y > 0.0f) ? val.y : alpha * (expf(val.y) - 1.0f);
        result.z = (val.z > 0.0f) ? val.z : alpha * (expf(val.z) - 1.0f);
        result.w = (val.w > 0.0f) ? val.w : alpha * (expf(val.w) - 1.0f);
        
        out[idx] = result;
        idx += stride;
    }
}

torch::Tensor elu_cuda(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    
    int n = x.numel();
    int n4 = n / 4;
    
    // Use 128 threads (4 warps) per block
    const int threads = 128;
    const int blocks = min(256, (n4 + threads - 1) / threads);
    
    elu_kernel_coalesced<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        alpha,
        n4
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda, "ELU activation with coalesced memory access (CUDA)");
}