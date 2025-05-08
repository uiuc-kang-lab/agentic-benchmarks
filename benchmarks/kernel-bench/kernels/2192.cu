#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile dimensions
#define TILE_M 16  // Tile size in the M dimension (output row)
#define TILE_N 16  // Tile size in the N dimension (output column)
#define BLOCK_K 32 // Chunk size along the K dimension

__global__ void balancedWorkloadKernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int K,
                                        int M,
                                        int N) {
    // Each block computes a tile of output C of size TILE_M x TILE_N
    int row = blockIdx.x * TILE_M + threadIdx.x; // Corresponds to output index i
    int col = blockIdx.y * TILE_N + threadIdx.y; // Corresponds to output index j

    float value = 0.0f;

    // Allocate shared memory for a tile of A and B for the current k-chunk
    __shared__ float As[BLOCK_K][TILE_M];
    __shared__ float Bs[BLOCK_K][TILE_N];

    // Loop over K in chunks of size BLOCK_K
    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Load a tile from A into shared memory.
        for (int t = threadIdx.x; t < BLOCK_K; t += TILE_M) {
            if (row < M && k0 + t < K) {
                As[t][threadIdx.x] = A[(k0 + t) * M + row];
            } else {
                As[t][threadIdx.x] = 0.0f;
            }
        }

        // Load a tile from B into shared memory.
        for (int t = threadIdx.y; t < BLOCK_K; t += TILE_N) {
            if (col < N && k0 + t < K) {
                Bs[t][threadIdx.y] = B[(k0 + t) * N + col];
            } else {
                Bs[t][threadIdx.y] = 0.0f;
            }
        }

        __syncthreads();

        // Compute partial dot product for this k-chunk
        for (int t = 0; t < BLOCK_K; ++t) {
            value += As[t][threadIdx.x] * Bs[t][threadIdx.y];
        }

        __syncthreads();
    }

    // Write the computed value to the output matrix C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function exposed to PyTorch

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // Dimensions: A: (K, M) and B: (K, N). We compute C = A.T * B, so C has shape (M, N).
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define block dimensions: each block computes a TILE_M x TILE_N tile of C
    dim3 block(TILE_M, TILE_N);
    // Grid covers the M and N dimensions
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    balancedWorkloadKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B with balanced workload (CUDA)");
}
