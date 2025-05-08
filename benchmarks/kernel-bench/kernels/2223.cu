#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel for computing C = A.T * B using grid-stride loops to handle larger workloads.
// A: shape (K, M), B: shape (K, N), C: shape (M, N)
// Each thread computes multiple elements of C by striding across the output matrix.
__global__ void stridedMatmulKernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int K,
                                      int M,
                                      int N) {
    // Calculate initial output indices based on block and thread indices.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Compute strides to jump to the next element that this thread should process
    int rowStride = blockDim.y * gridDim.y;
    int colStride = blockDim.x * gridDim.x;

    // Loop over output rows and columns using grid-stride loops
    for (int i = row; i < M; i += rowStride) {
        for (int j = col; j < N; j += colStride) {
            float sum = 0.0f;
            // Perform the dot-product over the K dimension.
            // A is stored as (K, M): element A[k][i] is at A[k * M + i]
            // B is stored as (K, N): element B[k][j] is at B[k * N + j]
            for (int k = 0; k < K; ++k) {
                sum += A[k * M + i] * B[k * N + j];
            }
            // Write the result to C: row-major layout (M, N): C[i][j] = C[i * N + j]
            C[i * N + j] = sum;
        }
    }
}

// The forward function exposed via PyBind11.
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

    // Set block size. Using a 16x16 block covers 256 threads per block.
    const int blockDimX = 16;
    const int blockDimY = 16;
    dim3 blockDim(blockDimX, blockDimY);

    // Compute grid dimensions so each thread can cover multiple output elements via stride loops.
    dim3 gridDim((N + blockDimX - 1) / blockDimX,
                 (M + blockDimY - 1) / blockDimY);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    stridedMatmulKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using grid-stride loops (CUDA)");
}
