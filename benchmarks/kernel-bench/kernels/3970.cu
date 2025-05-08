#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_main_kernel(const float4* __restrict__ x4,
                                    float4* __restrict__ out4,
                                    int num_vectors) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    
    while (idx < num_vectors) {
        float4 vec = x4[idx];
        float4 res;
        
        res.x = __fdividef(vec.x, 1.0f + fabsf(vec.x));
        res.y = __fdividef(vec.y, 1.0f + fabsf(vec.y));
        res.z = __fdividef(vec.z, 1.0f + fabsf(vec.z));
        res.w = __fdividef(vec.w, 1.0f + fabsf(vec.w));
        
        out4[idx] = res;
        idx += stride;
    }
}

__global__ void softsign_remainder_kernel(const float* __restrict__ x,
                                         float* __restrict__ out,
                                         int start_idx,
                                         int total_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start_idx;
    if (idx < total_elements) {
        out[idx] = __fdividef(x[idx], 1.0f + fabsf(x[idx]));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    const int vectorized_size = num_elements / 4;
    const int remainder_start = vectorized_size * 4;
    const int remainder = num_elements % 4;

    // Main vectorized kernel
    if (vectorized_size > 0) {
        constexpr int BLOCKS = 256;  // Adjusted for better occupancy
        constexpr int THREADS = 128; // 4 warps per block
        
        softsign_main_kernel<<<BLOCKS, THREADS>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            vectorized_size
        );
    }

    // Remainder elements
    if (remainder > 0) {
        constexpr int R_THREADS = 64;
        const int r_blocks = (remainder + R_THREADS - 1) / R_THREADS;
        
        softsign_remainder_kernel<<<r_blocks, R_THREADS>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            remainder_start,
            num_elements
        );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign vectorized tuned (CUDA)");
}