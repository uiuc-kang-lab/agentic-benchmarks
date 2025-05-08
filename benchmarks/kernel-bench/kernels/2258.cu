#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel for computing C = A.T * B using stride loops to handle larger workloads.
// A: shape (K, M), B: shape (K, N), C: shape (M, N)
// Note: A.T(i,k) = A(k,i)
__global__ void matMulKernelStride(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int K,
                                   int M,
                                   int N) {
    // Compute initial indices based on block and thread indices
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    // Compute strides
    int rowStride = blockDim.x * gridDim.x;
    int colStride = blockDim.y * gridDim.y;

    // Loop over C's row and column indices using stride loops for boundary safety
    for (int i = row; i < M; i += rowStride) {
        for (int j = col; j < N; j += colStride) {
            float sum = 0.0f;
            // Compute dot product for C[i, j]
            for (int k = 0; k < K; ++k) {
                // A is stored as (K, M), so element (k, i) is A[k * M + i]
                // B is stored as (K, N), so element (k, j) is B[k * N + j]
                sum += A[k * M + i] * B[k * N + j];
            }
            // C is (M, N) row-major
            C[i * N + j] = sum;
        }
    }
}

// The forward function exposed via PyBind11. Computes C = A.T * B
// A: Tensor of shape (K, M) [CUDA, float32]
// B: Tensor of shape (K, N) [CUDA, float32]
// Returns: Tensor C of shape (M, N)

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Check that inputs are CUDA tensors and of type float32
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // Get dimensions: A is (K, M) and B is (K, N)
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N)
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define block and grid sizes
    const int BLOCK_SIZE = 32;  // Increased from 16 to 32 for better occupancy
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    
    // Calculate grid dimensions to exactly cover the output matrix
    // Limit maximum grid dimensions to avoid excessive oversubscription
    const int MAX_GRID_DIM = 65535;  // Maximum grid dimension for most GPUs
    int gridDimX = min((M + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRID_DIM);
    int gridDimY = min((N + BLOCK_SIZE - 1) / BLOCK_SIZE, MAX_GRID_DIM);
    dim3 gridDim(gridDimX, gridDimY);

    // Launch the CUDA kernel
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();
    
    matMulKernelStride<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using stride loops");
}
