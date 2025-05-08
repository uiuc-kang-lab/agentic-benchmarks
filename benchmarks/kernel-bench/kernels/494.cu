#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void hybridCoalescedVectorizedMultiplyKernel(const float* __restrict__ A,
                                                        float* __restrict__ C,
                                                        float s,
                                                        int64_t size)
{
    const int warpSize = 32;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int idx4 = tid * 4;
    
    // Coalesced and vectorized memory access for main chunks
    if (idx4 + 3 < size) {
        float4 a4 = __ldg(reinterpret_cast<const float4*>(&A[idx4]));
        a4.x *= s;
        a4.y *= s;
        a4.z *= s;
        a4.w *= s;
        *reinterpret_cast<float4*>(&C[idx4]) = a4;
    }
    // Handle remaining elements at the end of the array
    else if (idx4 < size) {
        #pragma unroll
        for (int i = 0; i < 4 && idx4 + i < size; i++) {
            C[idx4 + i] = A[idx4 + i] * s;
        }
    }
}

torch::Tensor hybrid_forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    const int threadsPerBlock = 256; // Optimal for both coalescing and vectorization
    const int blocks = (size + threadsPerBlock * 4 - 1) / (threadsPerBlock * 4);
    
    hybridCoalescedVectorizedMultiplyKernel<<<blocks, threadsPerBlock>>>(A.data_ptr<float>(),
                                                                         C.data_ptr<float>(),
                                                                         s,
                                                                         size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hybrid_forward", &hybrid_forward, "Hybrid Coalesced and Vectorized matrix-scalar multiplication");
}