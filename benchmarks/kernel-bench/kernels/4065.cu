#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define WARP_SIZE 32

__device__ __forceinline__ float warp_elu(float val, float alpha) {
    return (val > 0) ? val : alpha * (expf(val) - 1);
}

__global__ void elu_kernel_warp(const float4* __restrict__ x, float4* __restrict__ out, float alpha, int n4) {
    const unsigned int tid = threadIdx.x;
    const unsigned int wid = tid / WARP_SIZE;
    const unsigned int lane = tid % WARP_SIZE;
    const unsigned int warp_per_block = blockDim.x / WARP_SIZE;
    const unsigned int gid = (blockIdx.x * warp_per_block + wid) * WARP_SIZE + lane;
    
    // Shared memory for tiling
    __shared__ float4 s_data[256];  // Assuming blockDim.x = 256
    
    float4 result;
    
    if (gid < n4) {
        // Load data into shared memory
        s_data[tid] = __ldg(&x[gid]);
        __syncthreads();
        
        // Process data from shared memory
        float4 in_val = s_data[tid];
        result.x = warp_elu(in_val.x, alpha);
        result.y = warp_elu(in_val.y, alpha);
        result.z = warp_elu(in_val.z, alpha);
        result.w = warp_elu(in_val.w, alpha);
        
        // Write result back to global memory
        out[gid] = result;
    }
}

__global__ void elu_kernel_remainder(const float* __restrict__ x, float* __restrict__ out, float alpha, int start, int n) {
    const unsigned int tid = threadIdx.x;
    const unsigned int gid = blockIdx.x * blockDim.x + tid + start;
    
    if (gid < n) {
        float val = __ldg(&x[gid]);
        out[gid] = warp_elu(val, alpha);
    }
}

torch::Tensor elu_cuda_warp(torch::Tensor x, float alpha) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int n = x.numel();
    int n4 = n / 4;
    
    const int threads = 256;
    const int blocks = (n4 + (threads / WARP_SIZE) - 1) / (threads / WARP_SIZE);
    
    if (n4 > 0) {
        elu_kernel_warp<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            alpha,
            n4
        );
    }
    
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
    m.def("forward", &elu_cuda_warp, "ELU activation with warp primitives (CUDA)");
}