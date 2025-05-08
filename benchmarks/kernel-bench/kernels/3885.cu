#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel_uniform(const float4* __restrict__ in, float4* __restrict__ out, int num_vector_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time using float4
    for (int idx = tid; idx < num_vector_elements; idx += stride) {
        float4 input = in[idx];
        
        // Process all components uniformly without branches
        float4 result;
        result.x = input.x / (1.0f + fabsf(input.x));
        result.y = input.y / (1.0f + fabsf(input.y));
        result.z = input.z / (1.0f + fabsf(input.z));
        result.w = input.w / (1.0f + fabsf(input.w));
        
        out[idx] = result;
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    int vector_elements = num_elements / 4;
    
    // Ensure alignment for float4
    TORCH_CHECK(num_elements % 4 == 0, "Input tensor size must be multiple of 4");
    
    // Configure kernel launch parameters
    const int threads = 256;
    const int blocks = std::min(65535, (vector_elements + threads - 1) / threads);
    
    softsign_kernel_uniform<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        vector_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-uniform Softsign activation (CUDA)");
}