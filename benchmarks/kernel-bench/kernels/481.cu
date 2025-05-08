#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__constant__ float d_scalar;

__global__ void constantMemMultiplyKernel(const float* __restrict__ A,
                                         float* __restrict__ C,
                                         int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx4 = idx * 4;
    
    // Process 4 elements at a time using float4
    if (idx4 + 3 < size) {
        float4 a4 = *reinterpret_cast<const float4*>(&A[idx4]);
        float4 c4;
        c4.x = a4.x * d_scalar;
        c4.y = a4.y * d_scalar;
        c4.z = a4.z * d_scalar;
        c4.w = a4.w * d_scalar;
        
        *reinterpret_cast<float4*>(&C[idx4]) = c4;
    }
    // Handle remaining elements
    else if (idx4 < size) {
        for (int i = 0; i < 4 && idx4 + i < size; i++) {
            C[idx4 + i] = __ldg(&A[idx4 + i]) * d_scalar;
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s)
{
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    
    // Copy scalar to constant memory
    cudaMemcpyToSymbol(d_scalar, &s, sizeof(float));
    
    const int threads = 256;
    const int blocks = ((size + 3) / 4 + threads - 1) / threads;
    
    constantMemMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                  C.data_ptr<float>(),
                                                  size);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Constant memory vectorized matrix-scalar multiplication kernel");
}