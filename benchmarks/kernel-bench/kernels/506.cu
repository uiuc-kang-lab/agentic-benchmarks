#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernel(const float* __restrict__ A,
                               float* __restrict__ C,
                               float s,
                               int64_t size)
{
    // Ensure 128-bit aligned access by using vectorized loads
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Process 4 elements at a time for 128-bit alignment
    int aligned_size = size & ~3;  // Round down to multiple of 4
    
    if (idx * 4 < aligned_size) {
        // Use vectorized load for 4 elements at once
        float4 a4;
        float4* a4_ptr = (float4*)&A[idx * 4];
        a4 = *reinterpret_cast<const float4*>(__ldg((const float4*)a4_ptr));
        
        // Perform scalar multiplication
        a4.x *= s;
        a4.y *= s;
        a4.z *= s;
        a4.w *= s;
        
        // Store result
        float4* c4_ptr = (float4*)&C[idx * 4];
        *c4_ptr = a4;
    }
    
    // Handle remaining elements
    if (idx < size && idx * 4 >= aligned_size) {
        C[idx] = __ldg(&A[idx]) * s;
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 256;
    const int blocks = (size + threads * 4 - 1) / (threads * 4);

    multiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                       C.data_ptr<float>(),
                                       s,
                                       size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix-scalar multiplication kernel");
}