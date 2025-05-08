#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Hybrid kernel: uses grid-stride loops and float4 vectorization for bulk data,
// with a fallback for leftover scalar elements.
__global__ void hybridMultiplyKernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       float s,
                                       int64_t n) {
    // Determine the number of full groups of 4 elements
    int fullGroups = n / 4;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    // Process full vectorized groups using grid-stride loop
    for (int i = tid; i < fullGroups; i += blockDim.x * gridDim.x) {
        // Cast pointer to float4 for vectorized access
        float4 a_val = __ldg(((const float4*)A) + i);
        float4 result;
        result.x = a_val.x * s;
        result.y = a_val.y * s;
        result.z = a_val.z * s;
        result.w = a_val.w * s;
        ((float4*)C)[i] = result;
    }
    
    // Process any remaining elements that don't form a complete float4
    int offset = fullGroups * 4;
    for (int i = tid; i < (n - offset); i += blockDim.x * gridDim.x) {
        C[offset + i] = __ldg(&A[offset + i]) * s;
    }
}

// The forward function performs input checks, prepares output tensor, and launches the kernel
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t n = A.numel();

    // Configure threads and blocks based on vectorized work: n/4 elements to process
    const int threads = 256;
    const int blocks = ((n / 4) + threads - 1) / threads;

    hybridMultiplyKernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        s,
        n
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid vectorized matrix-scalar multiplication kernel using grid-stride loops");
}
