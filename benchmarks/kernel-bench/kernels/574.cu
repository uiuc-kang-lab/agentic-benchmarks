#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void vectorizedMultiplyKernel(const float4* __restrict__ A,
                                        float4* __restrict__ C,
                                        float scalar,
                                        int64_t num_vectors) {
    const int vector_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    for(int i = vector_idx; i < num_vectors; i += stride) {
        float4 vec = A[i];
        vec.x *= scalar;
        vec.y *= scalar;
        vec.z *= scalar;
        vec.w *= scalar;
        C[i] = vec;
    }
}

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    const int64_t size = A.numel();
    const int64_t num_vectors = size / 4;
    const int64_t remainder = size % 4;

    // Main vectorized kernel
    if(num_vectors > 0) {
        const int threads = 256;
        const int blocks = (num_vectors + threads - 1) / threads;
        
        vectorizedMultiplyKernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(A.data_ptr<float>()),
            reinterpret_cast<float4*>(C.data_ptr<float>()),
            s,
            num_vectors
        );
    }

    // Handle remainder elements
    if(remainder > 0) {
        const int64_t start = num_vectors * 4;
        const int threads = 256;
        const int blocks = (remainder + threads - 1) / threads;
        
        __global__ void remainder_kernel(const float* A, float* C, float s, int size) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if(idx < size) C[idx] = A[idx] * s;
        };
        
        remainder_kernel<<<blocks, threads>>>(
            A.data_ptr<float>() + start,
            C.data_ptr<float>() + start,
            s,
            remainder
        );
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized vector4 matrix-scalar multiplication with perfect coalescing");
}