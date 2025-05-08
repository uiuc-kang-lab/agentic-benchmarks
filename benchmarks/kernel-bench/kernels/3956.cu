#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel_vectorized(const float4* x4, float4* out4, int num_vector_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_vector_elements) {
        float4 in = x4[idx];
        
        // Process all 4 elements
        float4 result;
        result.x = in.x / (1.0f + fabsf(in.x));
        result.y = in.y / (1.0f + fabsf(in.y));
        result.z = in.z / (1.0f + fabsf(in.z));
        result.w = in.w / (1.0f + fabsf(in.w));
        
        out4[idx] = result;
    }
}

__global__ void softsign_kernel_remainder(const float* x, float* out, int start_idx, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx + start_idx < num_elements) {
        float val = x[idx + start_idx];
        out[idx + start_idx] = val / (1.0f + fabsf(val));
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    int vector_elements = num_elements / 4;
    int remainder = num_elements % 4;
    
    const int threads = 256;
    int blocks = (vector_elements + threads - 1) / threads;
    
    if (vector_elements > 0) {
        softsign_kernel_vectorized<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(x.data_ptr<float>()),
            reinterpret_cast<float4*>(out.data_ptr<float>()),
            vector_elements
        );
    }
    
    // Handle remaining elements
    if (remainder > 0) {
        int remainder_blocks = (remainder + threads - 1) / threads;
        softsign_kernel_remainder<<<remainder_blocks, threads>>>(
            x.data_ptr<float>(),
            out.data_ptr<float>(),
            vector_elements * 4,
            num_elements
        );
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation vectorized (CUDA)");
}