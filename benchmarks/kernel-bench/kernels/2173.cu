#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel for computing C = A.T * B.
// A: shape (K, M), B: shape (K, N), C: shape (M, N)
// Note that A.T(i,k) = A(k,i)
__global__ void matMulKernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int K,
                             int M,
                             int N) {
    // Each thread computes one element of C.
    // i indexes rows of C (and columns of A), j indexes columns of C (and B).
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < M && j < N) {
        float sum = 0.0f;
        // Loop over k dimension of A (and B)
        for (int k = 0; k < K; ++k) {
            // A is stored as (K, M): element (k, i) is at A[k * M + i]
            // B is stored as (K, N): element (k, j) is at B[k * N + j]
            sum += A[k * M + i] * B[k * N + j];
        }
        // C is (M, N) with row-major storage: element (i, j) at C[i * N + j]
        C[i * N + j] = sum;
    }
}

// The forward function exposed via PyBind11.
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B.
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors and of type float32.
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // A: (K, M) and B: (K, N)
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N).
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define thread block and grid sizes.
    const int THREADS = 16;
    dim3 blockDim(THREADS, THREADS);
    dim3 gridDim((M + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS);

    // Get raw pointers to the data.
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the CUDA kernel.
    matMulKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA)");
}