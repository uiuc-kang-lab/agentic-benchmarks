#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_aligned_kernel(const float4* __restrict__ x4,
                                       float4* __restrict__ out4,
                                       int num_vectors) {
    constexpr int STRIDE = 128 / sizeof(float); // 128-bit alignment
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    #pragma unroll 2
    for(; idx < num_vectors; idx += gridDim.x * blockDim.x) {
        // Aligned vector load with read-only cache
        const float4 vec = __ldg(&x4[idx]);
        
        // Compute with fast math intrinsics
        float4 res;
        res.x = __fdividef(vec.x, 1.0f + fabsf(vec.x));
        res.y = __fdividef(vec.y, 1.0f + fabsf(vec.y));
        res.z = __fdividef(vec.z, 1.0f + fabsf(vec.z));
        res.w = __fdividef(vec.w, 1.0f + fabsf(vec.w));
        
        // Aligned vector store
        out4[idx] = res;
    }
}

__global__ void softsign_aligned_remainder(const float* __restrict__ x,
                                          float* __restrict__ out,
                                          int start,
                                          int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;
    if (idx < total) {
        // Scalar load with read-only cache
        const float val = __ldg(&x[idx]);
        out[idx] = __fdividef(val, 1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num = x.numel();
    const int vec_size = num / 4;
    const int rem_start = vec_size * 4;
    
    if (vec_size > 0) {
        constexpr int BLOCKS = 1024; // Maximize SM occupancy
        constexpr int THREADS = 128;
        
        softsign_aligned_kernel<<<BLOCKS, THREADS>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            vec_size
        );
    }
    
    if (num - rem_start > 0) {
        constexpr int R_THREADS = 64;
        const int r_blocks = (num - rem_start + R_THREADS - 1) / R_THREADS;
        
        softsign_aligned_remainder<<<r_blocks, R_THREADS>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            rem_start,
            num
        );
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign with aligned 128-bit LDG (CUDA)");
}