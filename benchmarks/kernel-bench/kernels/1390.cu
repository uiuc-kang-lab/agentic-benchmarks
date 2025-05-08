#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel: optimized for memory coalescing
__global__ void diag_matmul_kernel_coalesced(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N) {
        float a_val = A[row];
        for (int col = 0; col < M; ++col) {
            int idx = row * M + col;
            C[idx] = a_val * B[idx];
        }
    }
}

// Forward function that wraps our CUDA kernel
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0),
                "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure inputs are on contiguous memory
    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    // Create an output tensor with the same device and type as B
    auto C = torch::empty({N, M}, B.options());

    // Configure and launch the kernel
    const int64_t threads = 256;
    const int64_t blocks = (N + threads - 1) / threads;
    diag_matmul_kernel_coalesced<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

// Create the PyTorch extension module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication of A and B on the GPU");
}