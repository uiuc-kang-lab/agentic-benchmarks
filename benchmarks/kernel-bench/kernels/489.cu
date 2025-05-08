#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// This kernel processes the input in two uniform phases to avoid warp divergence:
// 1. Vectorized processing of full groups of 4 elements using float4 loads and stores.
// 2. Scalar processing of any remaining tail elements.
// Both phases use grid-stride loops so that all threads follow the same control flow,
// minimizing divergent branches within warps.

__global__ void combinedUniformMultiplyKernel(const float* __restrict__ A,
                                                 float* __restrict__ C,
                                                 float s,
                                                 int64_t n) {
    int stride = blockDim.x * gridDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute number of full groups of 4 elements
    int fullGroups = n / 4;
    int tailStart = fullGroups * 4;
    
    // Phase 1: Uniform vectorized processing using float4
    for (int i = tid; i < fullGroups; i += stride) {
        float4 a_val = reinterpret_cast<const float4*>(A)[i];
        float4 res;
        res.x = a_val.x * s;
        res.y = a_val.y * s;
        res.z = a_val.z * s;
        res.w = a_val.w * s;
        reinterpret_cast<float4*>(C)[i] = res;
    }
    
    // Phase 2: Uniform scalar processing for the remaining elements
    for (int i = tailStart + tid; i < n; i += stride) {
        C[i] = A[i] * s;
    }
}

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");
    
    auto C = torch::empty_like(A);
    int64_t n = A.numel();

    // Choose thread and block configuration appropriate for grid-stride loops
    const int threads = 256;
    int blocks = ((n / 4) + threads - 1) / threads;
    if (blocks < 1) blocks = 1;

    combinedUniformMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                         C.data_ptr<float>(),
                                                         s,
                                                         n);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined uniform without divergence matrix-scalar multiplication kernel");
}
