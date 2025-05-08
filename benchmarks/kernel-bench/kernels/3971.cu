#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_vector_kernel(const float4* __restrict__ x,
                                      float4* __restrict__ out,
                                      int num_vectors) {
    const int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    
    if (idx < num_vectors) {
        float4 vec = x[idx];
        
        float4 result;
        result.x = __fdividef(vec.x, 1.0f + fabsf(vec.x));
        result.y = __fdividef(vec.y, 1.0f + fabsf(vec.y));
        result.z = __fdividef(vec.z, 1.0f + fabsf(vec.z));
        result.w = __fdividef(vec.w, 1.0f + fabsf(vec.w));
        
        out[idx] = result;
    }
}

__global__ void softsign_tail_kernel(const float* __restrict__ x,
                                   float* __restrict__ out,
                                   int start_idx, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + start_idx < num_elements) {
        float val = x[start_idx + idx];
        out[start_idx + idx] = __fdividef(val, 1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    const int num_elements = x.numel();
    const int num_vectors = num_elements / 4;
    const int remainder = num_elements % 4;

    // Process aligned vectors
    if (num_vectors > 0) {
        const int threads = 256;
        const int blocks = (num_vectors + threads - 1) / threads;
        softsign_vector_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            num_vectors
        );
    }

    // Process remaining elements
    if (remainder > 0) {
        const int threads = 128;
        const int start_idx = num_vectors * 4;
        const int elements_remain = num_elements - start_idx;
        const int blocks = (elements_remain + threads - 1) / threads;
        softsign_tail_kernel<<<blocks, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            start_idx,
            num_elements
        );
    }

    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign optimized with vectorized unrolling");
}
