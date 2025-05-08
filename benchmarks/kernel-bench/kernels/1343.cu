#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses a grid-stride loop to handle workloads larger than the number of available threads.
// It computes C[i, j] = A[i] * B[i, j] by mapping each thread to multiple elements using the stride loop.
__global__ void stride_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int total = N * M;
    // Compute a unique index for this thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Use a grid-stride loop to cover all output elements
    for (; idx < total; idx += stride) {
        int row = idx / M;  // Compute row index
        // Multiply the corresponding diagonal value with the element from B
        C[idx] = A[row] * B[idx];
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    const int64_t N = A.size(0);
    const int64_t M = B.size(1);
    int64_t total = N * M;

    // Allocate output tensor
    auto C = torch::empty({N, M}, B.options());

    // Launch kernel with grid-stride loop to cover all elements
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    stride_diag_matmul_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride loop based diagonal matrix multiplication");
}
