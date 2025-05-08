#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel for computing C = A.T * B using grid-stride loops and loop unrolling.
// A: shape (K, M) stored in row-major order for each k (A[k, i] is at A[k * M + i])
// B: shape (K, N) stored in row-major order (B[k, j] is at B[k * N + j])
// C: shape (M, N) stored in row-major order (C[i, j] is at C[i * N + j])
// The operation computed is: C(i,j) = sum_{k=0}^{K-1} A[k * M + i] * B[k * N + j]

__global__ void gridStrideUnrollKernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int K,
                                        int M,
                                        int N) {
    // Calculate starting indices for output matrix using 2D grid-stride loops
    int i_start = blockIdx.x * blockDim.x + threadIdx.x;
    int j_start = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Compute strides for grid iteration
    int stride_i = gridDim.x * blockDim.x;
    int stride_j = gridDim.y * blockDim.y;
    
    // Loop over output indices (i, j) with proper boundary checks
    for (int i = i_start; i < M; i += stride_i) {
        for (int j = j_start; j < N; j += stride_j) {
            float sum = 0.0f;
            
            // Unroll the loop over K by a factor of 4 for better ILP
            int k = 0;
            int K4 = (K / 4) * 4;
            for (; k < K4; k += 4) {
                sum += A[(k) * M + i]     * B[(k) * N + j] +
                       A[(k+1) * M + i]   * B[(k+1) * N + j] +
                       A[(k+2) * M + i]   * B[(k+2) * N + j] +
                       A[(k+3) * M + i]   * B[(k+3) * N + j];
            }
            // Handle the remaining elements if K is not a multiple of 4
            for (; k < K; ++k) {
                sum += A[k * M + i] * B[k * N + j];
            }
            
            C[i * N + j] = sum;
        }
    }
}

// The forward function exposed via PyBind11.
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N), computed as A.T * B.

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

    // Use a 2D block and grid with grid-stride loops handling larger workloads
    dim3 blockDim(16, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    gridStrideUnrollKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using grid-stride loops with unrolling (CUDA)");
}
