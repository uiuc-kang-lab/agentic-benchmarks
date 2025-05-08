#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using grid-stride loops, vectorized loads via float4, and __ldg() for optimized read-only global memory access.
__global__ void elu_kernel_ldg(const float4* __restrict__ x, float4* __restrict__ out, float alpha, int n4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridStride = gridDim.x * blockDim.x;
    
    for (; idx < n4; idx += gridStride) {
        // Use __ldg() for read-only access to global memory, aligned to 128 bits
        float4 in = __ldg(&x[idx]);
        float4 result;
        result.x = (in.x > 0.f) ? in.x : alpha * (expf(in.x) - 1.f);
        result.y = (in.y > 0.f) ? in.y : alpha * (expf(in.y) - 1.f);
        result.z = (in.z > 0.f) ? in.z : alpha * (expf(in.z) - 1.f);
        result.w = (in.w > 0.f) ? in.w : alpha * (expf(in.w) - 1.f);
        out[idx] = result;
    }
}

// Kernel to handle remaining elements that don't fit into a multiple of 4
__global__ void elu_kernel_ldg_remainder(const float* __restrict__ x, float* __restrict__ out, float alpha, int offset, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    int gridStride = gridDim.x * blockDim.x;
    
    for (; idx < n; idx += gridStride) {
        float val = __ldg(&x[idx]);
        out[idx] = (val > 0.f) ? val : alpha * (expf(val) - 1.f);
    }
}

// Host function that wraps the CUDA kernels
torch::Tensor elu_cuda_ldg(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    auto out = torch::empty_like(x);
    
    int n = x.numel();
    int n4 = n / 4;         // Number of 4-element groups
    int remainder = n % 4;    // Remaining elements
    
    const int threads = 256;
    
    if(n4 > 0) {
        int blocks = (n4 + threads - 1) / threads;
        elu_kernel_ldg<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            n4
        );
    }
    
    if(remainder > 0) {
        int offset = n4 * 4;
        int blocks_rem = (remainder + threads - 1) / threads;
        elu_kernel_ldg_remainder<<<blocks_rem, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            alpha,
            offset,
            n
        );
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &elu_cuda_ldg, "ELU activation using __ldg and 128-bit aligned accesses (CUDA)");
}
