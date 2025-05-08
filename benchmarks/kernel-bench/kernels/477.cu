#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void optimizedVectorMultiply(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int64_t size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int vector_idx = idx * 4;

    // Process 4 elements with coalesced memory access
    if (vector_idx + 3 < size) {
        float4 a_chunk = *reinterpret_cast<const float4*>(A + vector_idx);
        float4 c_chunk;
        c_chunk.x = a_chunk.x * s;
        c_chunk.y = a_chunk.y * s;
        c_chunk.z = a_chunk.z * s;
        c_chunk.w = a_chunk.w * s;
        *reinterpret_cast<float4*>(C + vector_idx) = c_chunk;
    }
    // Handle remaining elements with scalar operations
    else {
        for (int i = 0; i < 4; ++i) {
            if (vector_idx + i < size) {
                C[vector_idx + i] = __ldg(&A[vector_idx + i]) * s;
            }
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
    const int elements_per_thread = 4;
    const int blocks = (size + elements_per_thread * threads - 1) / (elements_per_thread * threads);

    optimizedVectorMultiply<<<blocks, threads>>>(A.data_ptr<float>(),
                                               C.data_ptr<float>(),
                                               s,
                                               size);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized vectorized matrix-scalar multiplication");
}