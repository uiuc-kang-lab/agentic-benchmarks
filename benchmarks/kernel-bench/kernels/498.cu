#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

// Kernel that manually unrolls the inner loop to reduce loop overhead and improve performance
__global__ void manualUnrolledMultiplyKernel(const float* __restrict__ A,
                                             float* __restrict__ C,
                                             float s,
                                             int n) {
    // Each thread processes four elements in a contiguous block determined by a fixed unroll factor
    int idx = blockIdx.x * blockDim.x * 4 + threadIdx.x;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        int pos = idx + i * blockDim.x;
        if (pos < n) {
            C[pos] = A[pos] * s;
        }
    }
}

// Forward function called from Python
torch::Tensor forward(torch::Tensor A, float s) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor.");
    TORCH_CHECK(A.scalar_type() == torch::kFloat, "Input tensor A must be of type float.");

    auto C = torch::empty_like(A);
    int n = A.numel();
    
    // Each block processes (threads * 4) elements due to unrolling factor of 4
    const int threads = 256;
    const int blocks = (n + threads * 4 - 1) / (threads * 4);

    manualUnrolledMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                      C.data_ptr<float>(),
                                                      s,
                                                      n);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Manual unrolled matrix-scalar multiplication kernel");
}
