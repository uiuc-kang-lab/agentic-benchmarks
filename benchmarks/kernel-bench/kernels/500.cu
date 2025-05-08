#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void alignedLdgMemoryAccessKernel(const float* __restrict__ A,
                                              float* __restrict__ C,
                                              float s,
                                              int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx4 = idx * 4;
    
    // Process 4 elements at a time using float4
    if (idx4 + 3 < size) {
        float4 a4 = __ldg((const float4*)(&A[idx4]));
        float4 result;
        result.x = a4.x * s;
        result.y = a4.y * s;
        result.z = a4.z * s;
        result.w = a4.w * s;
        ((float4*)C)[idx] = result;
    }
    // Handle remaining elements
    else if (idx < (size + 3) / 4) {
        int base = idx4;
        for (int i = 0; i < 4 && base + i < size; i++) {
            C[base + i] = __ldg(&A[base + i]) * s;
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    const int threads = 256;
    const int blocks = ((size + 3) / 4 + threads - 1) / threads;
    
    alignedLdgMemoryAccessKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                      C.data_ptr<float>(),
                                                      s,
                                                      size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Aligned __ldg memory access matrix-scalar multiplication kernel");
}