/*
 * This CUDA kernel computes C = A.T * B, where A has shape (K, M) and B has shape (K, N).
 * Effectively, with A^T having shape (M, K), the multiplication is
 *   C[i,j] = sum_{k=0}^{K-1} A[k, i] * B[k, j]
 * We use tiling and shared memory to load chunks of A and B, avoiding costly atomic adds
 * by assigning each output tile exclusively to one thread block.
 *
 * Note: A is stored in row-major order such that element A(k, i) is at index k*M + i.
 *       Similarly, B(k, j) is at index k*N + j, and C(i,j) is stored at C[i*N + j].
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile size for both output dimensions and the K-dimension tiling
#define TILE_SIZE 16

// The tiled shared memory kernel computes one tile of C per block.
// Each block is 2D with dimensions (TILE_SIZE, TILE_SIZE).
// The kernel loads tiles of the effective A and B matrices into shared memory.
// The effective matrix A_effective is defined as A^T with A_effective[i, k] = A[k*M + i].
// This kernel avoids atomic adds by performing a complete accumulation before writing to C.
__global__ void tiledSharedKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int K,
                                  int M,
                                  int N) {
    // Compute row (i) and column (j) indices in the output matrix C
    int row = blockIdx.x * TILE_SIZE + threadIdx.x;  // corresponds to index i in C
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;  // corresponds to index j in C

    float sum = 0.0f;

    // Declare shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over tiles in the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Each thread loads one element of the A tile into shared memory
        // For effective A: A_effective[row, k] = A[k*M + row] with k = t*TILE_SIZE + threadIdx.y
        int a_k = t * TILE_SIZE + threadIdx.y;
        if (row < M && a_k < K) {
            As[threadIdx.x][threadIdx.y] = A[a_k * M + row];
        } else {
            As[threadIdx.x][threadIdx.y] = 0.0f;
        }

        // Each thread loads one element of the B tile into shared memory
        // B is stored such that B[k, col] = B[k*N + col] with k = t*TILE_SIZE + threadIdx.x
        int b_k = t * TILE_SIZE + threadIdx.x;
        if (col < N && b_k < K) {
            Bs[threadIdx.x][threadIdx.y] = B[b_k * N + col];
        } else {
            Bs[threadIdx.x][threadIdx.y] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles and accumulate the result
        // Each thread computes a partial sum for C[row, col]
        for (int k_idx = 0; k_idx < TILE_SIZE; k_idx++) {
            sum += As[threadIdx.x][k_idx] * Bs[k_idx][threadIdx.y];
        }

        __syncthreads();
    }

    // Write the computed value to C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11.
// It verifies the input tensor types and shapes, allocates the output tensor, and launches the kernel.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors of type float32
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // A: shape (K, M) and B: shape (K, N).
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N) and initialize to zeros
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define block dimensions (each block computes a TILE_SIZE x TILE_SIZE output tile)
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    // Grid dimensions cover the entire output matrix C
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the CUDA kernel
    tiledSharedKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using tiled shared memory kernel (CUDA)");
}
