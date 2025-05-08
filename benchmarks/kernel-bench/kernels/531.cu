#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Optimized kernel that combines vectorized loads with loop unrolling
__global__ void multiplyKernelOptimized(const float* __restrict__ A,
                                          float* __restrict__ C,
                                          float s,
                                          int64_t size) {
    // Each thread processes 16 elements (4 groups of 4 floats)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int base = idx * 16;

    // Unroll 4 iterations, each processing 4 contiguous floats via vectorized load
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int pos = base + i * 4;
        // If we have 4 elements available, use 128-bit (float4) vectorized load/store
        if (pos + 3 < size) {
            // Load 4 floats from global memory using __ldg for read-only caching
            float4 a_val = __ldg(reinterpret_cast<const float4*>(&A[pos]));
            a_val.x = __fmulf_rn(a_val.x, s);
            a_val.y = __fmulf_rn(a_val.y, s);
            a_val.z = __fmulf_rn(a_val.z, s);
            a_val.w = __fmulf_rn(a_val.w, s);
            // Store the result back
            *reinterpret_cast<float4*>(&C[pos]) = a_val;
        } else if (pos < size) {
            // Fallback for tail elements: process any remaining elements scalarly
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                int cur = pos + j;
                if (cur < size) {
                    C[cur] = A[cur] * s;
                }
            }
        }
    }
}

// C++ interface for PyTorch
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t size = A.numel();
    const int threads = 256;
    // Each thread processes 16 elements
    const int blocks = (size + threads * 16 - 1) / (threads * 16);

    multiplyKernelOptimized<<<blocks, threads>>>(A.data_ptr<float>(),
                                                   C.data_ptr<float>(),
                                                   s,
                                                   size);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix-scalar multiplication kernel combining vectorized loads and loop unrolling");
}
