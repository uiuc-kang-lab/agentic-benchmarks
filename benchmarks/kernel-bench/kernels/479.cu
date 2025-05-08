#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void unrolledVectorizedMultiplyKernel(const float* __restrict__ A,
                                                  float* __restrict__ C,
                                                  float s,
                                                  int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx4 = idx * 4;
    
    // Process 4 elements at a time using float4
    if (idx4 + 3 < size) {
        float4 a4;
        float4* a4_ptr = (float4*)(&A[idx4]);
        float4* c4_ptr = (float4*)(&C[idx4]);
        
        a4 = *a4_ptr;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            ((float*)&a4)[i] *= s;
        }
        
        *c4_ptr = a4;
    }
    // Handle remaining elements
    else if (idx < (size + 3) / 4) {
        int base = idx4;
        #pragma unroll
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
    
    unrolledVectorizedMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                         C.data_ptr<float>(),
                                                         s,
                                                         size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unrolled vectorized matrix-scalar multiplication kernel");
}