#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void optimizedHybridMultiplyKernel(const float* __restrict__ A,
                                               float* __restrict__ C,
                                               float s,
                                               int64_t n) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int fullGroups = n / 4;
    int offset = fullGroups * 4;

    // Process full vectorized groups using grid-stride loop
    for (int i = tid; i < fullGroups; i += blockDim.x * gridDim.x) {
        float4 a_val = __ldg(((const float4*)A) + i);
        float4 result;
        result.x = a_val.x * s;
        result.y = a_val.y * s;
        result.z = a_val.z * s;
        result.w = a_val.w * s;
        ((float4*)C)[i] = result;
    }

    // Process any remaining elements that don't form a complete float4
    for (int i = tid * 4; i < n; i += blockDim.x * gridDim.x * 4) {
        if (i < offset) continue; // Skip already processed full groups
        for (int j = 0; j < 4 && i + j < n; ++j) {
            C[i + j] = __ldg(&A[i + j]) * s;
        }
    }
}

torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t n = A.numel();

    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);

    optimizedHybridMultiplyKernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        s,
        n
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized hybrid vectorized matrix-scalar multiplication kernel");
}