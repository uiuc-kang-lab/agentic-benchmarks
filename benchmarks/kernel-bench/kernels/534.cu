#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void optimizedMultiplyKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int64_t size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Process 8 elements at a time using float4 and loop unrolling
    int aligned_size = size & ~7;  // Round down to multiple of 8
    
    for (int idx = tid * 8; idx < aligned_size; idx += stride * 8) {
        // Load two float4s (8 elements total)
        float4 a4_1 = __ldg((float4*)&A[idx]);
        float4 a4_2 = __ldg((float4*)&A[idx + 4]);
        
        // Multiply each component
        a4_1.x *= s;
        a4_1.y *= s;
        a4_1.z *= s;
        a4_1.w *= s;
        
        a4_2.x *= s;
        a4_2.y *= s;
        a4_2.z *= s;
        a4_2.w *= s;
        
        // Store results
        ((float4*)&C[idx])[0] = a4_1;
        ((float4*)&C[idx + 4])[0] = a4_2;
    }
    
    // Handle remaining elements
    for (int idx = tid + aligned_size; idx < size; idx += stride) {
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
    const int blocks = min(65535, static_cast<int>((size + threads * 8 - 1) / (threads * 8)));

    optimizedMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                C.data_ptr<float>(),
                                                s,
                                                size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix-scalar multiplication kernel");
}