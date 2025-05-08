#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

template<typename T>
__device__ __inline__ T* align_ptr(T* ptr) {
    return (T*)(((uintptr_t)ptr + 15) & ~15);
}

__global__ void softsign_2d_kernel(const float* __restrict__ x,
                                  float* __restrict__ out,
                                  int num_elements) {
    // 2D block layout for coalesced memory patterns
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Align pointer for vectorized access
    const float4* x4 = reinterpret_cast<const float4*>(__builtin_assume_aligned(x, 16));
    float4* out4 = reinterpret_cast<float4*>(__builtin_assume_aligned(out, 16));
    
    // Combine dimensions for grid-stride loop
    int tid = idx + idy * blockDim.x * gridDim.x;
    const int stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
    
    // Process elements with 4-wide vectorization first
    int vec_elements = num_elements / 4;
    while (tid < vec_elements) {
        float4 val = x4[tid];
        float4 res;
        
        res.x = __fdividef(val.x, 1.0f + fabsf(val.x));
        res.y = __fdividef(val.y, 1.0f + fabsf(val.y));
        res.z = __fdividef(val.z, 1.0f + fabsf(val.z));
        res.w = __fdividef(val.w, 1.0f + fabsf(val.w));
        
        out4[tid] = res;
        tid += stride;
    }
    
    // Handle remaining scalar elements
    tid = idx + idy * blockDim.x * gridDim.x + (vec_elements * 4);
    while (tid < num_elements) {
        float val = x[tid];
        out[tid] = __fdividef(val, 1.0f + fabsf(val));
        tid += stride;
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    
    // Optimized 2D grid config for H100
    const dim3 threads(32, 8);  // 256 threads/block in 2D
    const dim3 blocks(
        (num_elements + threads.x - 1) / threads.x,
        (144 * 4)  // Fill SMs with y-dimension blocks (144 SMs * 4)
    );
    
    softsign_2d_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        num_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign 2D optimized (CUDA)");
}
