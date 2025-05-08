#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void elu_kernel_vectorized(const float4* x, float4* out, float alpha, int n4) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n4) {
        float4 in = x[tid];
        float4 result;
        
        // Process four elements at once
        result.x = (in.x > 0) ? in.x : alpha * (expf(in.x) - 1);
        result.y = (in.y > 0) ? in.y : alpha * (expf(in.y) - 1);
        result.z = (in.z > 0) ? in.z : alpha * (expf(in.z) - 1);
        result.w = (in.w > 0) ? in.w : alpha * (expf(in.w) - 1);
        
        out[tid] = result;
    }
}

// Handle remaining elements
__global__ void elu_kernel_remainder(const float* x, float* out, float alpha, int start, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = start + tid;
    
    if (idx < n) {
        float val = x[idx];
        out[idx] = (val > 0) ? val : alpha * (expf(val) - 1);
    }
}

torch::Tensor elu_cuda_vectorized(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    int n4 = n / 4;
    
    const int threads = 512;
    const int blocks = (n4 + threads - 1) / threads;
    
    // Process blocks of 4 elements
    if (n4 > 0) {
        elu_kernel_vectorized<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            n4
        );
    }
    
    // Handle remaining elements
    int remaining = n - (n4 * 4);
    if (remaining > 0) {
        int remainder_blocks = (remaining + threads - 1) / threads;
        elu_kernel_remainder<<<remainder_blocks, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            alpha,
            n4 * 4,
            n
        );
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_vectorized, "ELU activation vectorized (CUDA)");
}