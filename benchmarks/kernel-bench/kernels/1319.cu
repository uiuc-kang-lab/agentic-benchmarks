#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = N * M;

    // Align to 128 bytes (32 floats) for coalesced memory access
    const int aligned_total = (total / 32) * 32;

    // Process aligned elements using coalesced memory access
    for (int idx = tid; idx < aligned_total; idx += stride) {
        const int row = idx / M;
        const float a_val = A[row];
        C[idx] = a_val * B[idx];
    }

    // Handle remaining elements
    for (int idx = aligned_total + tid; idx < total; idx += stride) {
        const int row = idx / M;
        C[idx] = A[row] * B[idx];
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    auto C = torch::empty({N, M}, B.options());

    const int threads = 128;
    const int blocks = min(65535, (int)((N * M + threads - 1) / threads));

    diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication with aligned coalesced memory access");
}