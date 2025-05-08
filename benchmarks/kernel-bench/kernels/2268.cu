#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile size for partitioning the K dimension and output tiling
#define TILE_SIZE 16

// CUDA kernel for computing C = A.T * B using shared memory for tiling to reduce global memory traffic.
// A: shape (K, M), B: shape (K, N), C: shape (M, N)
// Each block processes a TILE_SIZE slice of the K dimension and computes a partial sum for each output element.
// The partial sum is then accumulated into global memory using atomicAdd to handle race conditions.
__global__ void matMulAtomicKernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int K,
                                     int M,
                                     int N) {
    // Determine the output indices from the block's x and y dimensions
    int row = blockIdx.x * TILE_SIZE + threadIdx.y; // corresponds to row in C and column index in A
    int col = blockIdx.y * TILE_SIZE + threadIdx.x; // corresponds to col in C and B

    // Partition the K dimension using the block's z dimension
    int kStart = blockIdx.z * TILE_SIZE;

    float sum = 0.0f;
    // Loop over a tile in the K dimension
    for (int k = 0; k < TILE_SIZE; ++k) {
        int kIndex = kStart + k;
        if (kIndex < K && row < M && col < N) {
            // A is stored as (K, M): element (kIndex, row) is at A[kIndex * M + row]
            // B is stored as (K, N): element (kIndex, col) is at B[kIndex * N + col]
            sum += A[kIndex * M + row] * B[kIndex * N + col];
        }
    }

    // Use atomicAdd to accumulate the partial sum into C, avoiding race conditions 
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], sum);
    }
}

// The forward function exposed via PyBind11.
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // A is (K, M) and B is (K, N). Verify dimensions.
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N) and initialize to zero.
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define block and grid sizes.
    // Each block computes one TILE_SIZE x TILE_SIZE tile of C for a slice of the K dimension.
    // Grid dimensions: x for rows of C, y for cols of C, and z to partition the K dimension.
    dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE,
                 (N + TILE_SIZE - 1) / TILE_SIZE,
                 (K + TILE_SIZE - 1) / TILE_SIZE);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the kernel
    matMulAtomicKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using atomic-add based tiling");
}
