#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_DIAG_SIZE 16384

__constant__ float d_A[MAX_DIAG_SIZE];

__device__ __forceinline__ void compute_indices(int idx, int M, int* row, int* col) {
    *row = idx / M;
    *col = idx % M;
}

__device__ __forceinline__ float load_and_multiply(int row, int col, const float* B, int64_t M) {
    return d_A[row] * B[row * M + col];
}

__global__ void diag_matmul_kernel(
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * M) {
        int row, col;
        compute_indices(idx, M, &row, &col);
        C[idx] = load_and_multiply(row, col, B, M);
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch");
    TORCH_CHECK(A.size(0) <= MAX_DIAG_SIZE, "Diagonal too large");

    A = A.contiguous().cpu();
    B = B.contiguous();

    cudaMemcpyToSymbol(d_A, A.data_ptr<float>(), A.size(0) * sizeof(float));

    auto C = torch::empty_like(B);
    const int64_t total = A.size(0) * B.size(1);
    const int64_t threads = 256;
    diag_matmul_kernel<<<(total + threads - 1) / threads, threads>>>(
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        A.size(0),
        B.size(1)
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized diagonal matrix multiply");
}