#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void multiplyKernelUnroll(const float* __restrict__ A,
                               float* __restrict__ C,
                               float s,
                               int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process in chunks of 4 for vectorized memory operations
    int aligned_size = size & ~3;  // Round down to multiple of 4

    // Unroll loop for better performance
    for (int i = idx * 4; i < aligned_size; i += stride * 4) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float4 a4 = *reinterpret_cast<const float4*>(&A[i + j * stride]);
            a4.x *= s;
            a4.y *= s;
            a4.z *= s;
            a4.w *= s;
            *reinterpret_cast<float4*>(&C[i + j * stride]) = a4;
        }
    }

    // Handle remaining elements
    for (int i = aligned_size + idx; i < size; i += stride) {
        C[i] = A[i] * s;
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

    multiplyKernelUnroll<<<blocks, threads>>>(A.data_ptr<float>(),
                                       C.data_ptr<float>(),
                                       s,
                                       size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unroll optimized matrix-scalar multiplication kernel");
}