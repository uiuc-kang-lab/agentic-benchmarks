#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void softsign_kernel_vectorized(const float4* __restrict__ x4, float4* __restrict__ out4, int num_vector_elements) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < num_vector_elements) {
        float4 in = x4[tid];
        
        // Process all four elements
        float4 result;
        result.x = in.x / (1.0f + fabsf(in.x));
        result.y = in.y / (1.0f + fabsf(in.y));
        result.z = in.z / (1.0f + fabsf(in.z));
        result.w = in.w / (1.0f + fabsf(in.w));
        
        out4[tid] = result;
    }
}

torch::Tensor forward(torch::Tensor x) {
    CHECK_INPUT(x);
    
    auto out = torch::empty_like(x);
    int num_elements = x.numel();
    int vector_size = 4;
    int num_vector_elements = num_elements / vector_size;
    
    // Ensure the input size is a multiple of 4
    TORCH_CHECK(num_elements % vector_size == 0, "Input tensor size must be a multiple of 4");
    
    int threads = 256;  // Reduced thread count since each thread handles 4 elements
    int blocks = (num_vector_elements + threads - 1) / threads;
    
    softsign_kernel_vectorized<<<blocks, threads>>>(
        reinterpret_cast<const float4*>(x.data_ptr<float>()),
        reinterpret_cast<float4*>(out.data_ptr<float>()),
        num_vector_elements
    );
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Softsign activation vectorized (CUDA)");
}