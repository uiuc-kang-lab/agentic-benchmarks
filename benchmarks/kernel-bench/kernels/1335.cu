#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using grid-stride loop to cover all elements
__global__ void strided_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    const int total = N * M;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    while (idx < total) {
        int row = idx / M;  // Determine the row for index idx
        C[idx] = A[row] * B[idx];
        idx += stride;
    }
}

// Forward function: prepares tensors and launches the kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure contiguous memory
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    // Determine total number of elements and configure launch
    int threads = 256;
    int blocks = (N * M + threads - 1) / threads;

    // Launch the kernel using grid-stride loop for boundary handling
    strided_diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N, M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication with grid-stride loop for large workloads");
}
