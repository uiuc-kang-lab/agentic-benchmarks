#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__device__ __forceinline__ float4 elu_op(float4 val, float alpha) {
    float4 result;
    result.x = (val.x > 0) ? val.x : alpha * (expf(val.x) - 1);
    result.y = (val.y > 0) ? val.y : alpha * (expf(val.y) - 1);
    result.z = (val.z > 0) ? val.z : alpha * (expf(val.z) - 1);
    result.w = (val.w > 0) ? val.w : alpha * (expf(val.w) - 1);
    return result;
}

__global__ void elu_kernel_vectorized(const float* __restrict__ x, float* __restrict__ out, 
                                    float alpha, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time using float4
    int vec_n = n / 4;
    const float4* x4 = reinterpret_cast<const float4*>(x);
    float4* out4 = reinterpret_cast<float4*>(out);
    
    for (int i = idx; i < vec_n; i += stride) {
        float4 val = x4[i];
        out4[i] = elu_op(val, alpha);
    }
    
    // Handle remaining elements
    int remaining_start = vec_n * 4;
    for (int i = remaining_start + idx; i < n; i += stride) {
        float val = __ldg(&x[i]);
        out[i] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

torch::Tensor elu_cuda_vectorized(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    
    const int threads = 256;
    const int blocks = min((n + threads * 4 - 1) / (threads * 4), 65535);
    
    elu_kernel_vectorized<<<blocks, threads>>>(x.data_ptr<float>(), 
                                             out.data_ptr<float>(), 
                                             alpha, n);
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_vectorized, "ELU activation vectorized (CUDA)");
}