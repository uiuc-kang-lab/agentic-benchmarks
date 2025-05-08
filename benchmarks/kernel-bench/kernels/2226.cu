#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel for computing C = A.T * B using grid-stride loops.
// A: tensor of shape (K, M) where A is stored in row-major order (with K rows and M columns)
// B: tensor of shape (K, N) where B is stored in row-major order
// C: tensor of shape (M, N) computed as C[i, j] = sum_{k=0}^{K-1} A[k * M + i] * B[k * N + j]
__global__ void strideMatMulKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int K, int M, int N) {
    // Compute the initial row and column indices for C
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute the strides for the row and column dimensions
    int stride_row = gridDim.x * blockDim.x;
    int stride_col = gridDim.y * blockDim.y;

    // Loop over elements of C using grid-stride loops
    for (int i = row; i < M; i += stride_row) {
        for (int j = col; j < N; j += stride_col) {
            float sum = 0.0f;
            // Accumulate the dot product over K
            for (int k = 0; k < K; k += 4) {
                float4 a = make_float4(A[k * M + i], A[(k + 1) * M + i], A[(k + 2) * M + i], A[(k + 3) * M + i]);
                float4 b = make_float4(B[k * N + j], B[(k + 1) * N + j], B[(k + 2) * N + j], B[(k + 3) * N + j]);
                sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                // A is stored as (K, M) and B as (K, N)
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// The forward function exposed via PyBind11
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define thread block and grid sizes using 16x16 blocks
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    // The grid dimensions cover the output matrix, with additional iterations handled by stride loops
    dim3 gridDim((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    strideMatMulKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using grid-stride loops (CUDA)");
}
