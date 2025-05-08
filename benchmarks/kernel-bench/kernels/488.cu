#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// This kernel performs a matrix-scalar multiplication with vectorized memory accesses.
// Each thread covers unique indices, so no atomic operations are needed.
// The use of grid-stride loops and float4 loads/stores maximizes memory throughput while avoiding race conditions and global memory contention.

__global__ void vectorizedMultiplyKernel(const float* __restrict__ A,
                                           float* __restrict__ C,
                                           float s,
                                           int64_t n) {
    // Process groups of 4 floats at a time if possible
    int groupCount = n / 4;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Vectorized loop using float4; each thread handles multiple groups via grid-stride loop
    for (int i = tid; i < groupCount; i += stride) {
        float4 a_val = ((const float4*)A)[i];
        float4 c_val;
        c_val.x = a_val.x * s;
        c_val.y = a_val.y * s;
        c_val.z = a_val.z * s;
        c_val.w = a_val.w * s;
        ((float4*)C)[i] = c_val;
    }

    // Handle any remainder elements
    int remainStart = groupCount * 4;
    for (int j = remainStart + tid; j < n; j += stride) {
        // No race conditions occur here, so atomic operations are unnecessary
        C[j] = A[j] * s;
    }
}

// The forward function validates inputs, allocates output, and launches the kernel
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int64_t n = A.numel();
    
    const int threads = 256;
    const int blocks = (((n / 4) + threads - 1) / threads);

    vectorizedMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                    C.data_ptr<float>(),
                                                    s,
                                                    n);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Atomic free vectorized matrix-scalar multiplication kernel");
}
