#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile sizes and K-chunk size
#define TILE_M 16
#define TILE_N 16
#define BLOCK_K 32

// This kernel partitions the K-dimension into chunks using the z-dimension of the grid.
// Each block computes a tile of the output matrix C over a subset of the K dimension.
// For a given output element C(i,j), each thread computes a partial sum over its assigned chunk
// and then uses atomicAdd to accumulate the partial result. Atomic operations are thus used only
// once per thread per block, minimizing contention in global memory.
__global__ void tiledAtomicKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int K,
                                    int M,
                                    int N) {
    // Compute global output indices i and j
    int i = blockIdx.x * TILE_M + threadIdx.x;
    int j = blockIdx.y * TILE_N + threadIdx.y;
    if (i >= M || j >= N) return;

    float sum = 0.0f;
    // Determine the K-chunk this block is responsible for
    int k_start = blockIdx.z * BLOCK_K;
    int k_end = min(K, k_start + BLOCK_K);

    // Accumulate partial sum over the assigned K-chunk
    for (int k = k_start; k < k_end; k++) {
        // A is stored as (K, M): element A(k, i) at A[k*M + i]
        // B is stored as (K, N): element B(k, j) at B[k*N + j]
        sum += A[k * M + i] * B[k * N + j];
    }
    
    // Use atomicAdd to safely accumulate the partial sum into global memory
    // This is necessary because multiple blocks (over the z-dimension) contribute to the same C(i,j)
    atomicAdd(&C[i * N + j], sum);
}

// The forward function exposed via PyBind11. It sets up the grid to partition the operation
// across the M, N, and K dimensions using a 3D grid. This design minimizes the number of atomic adds
// by ensuring that each thread writes only one atomic addition per output element per K-chunk.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Checks for CUDA and float32 data
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // Dimensions: A: (K, M) and B: (K, N) 
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N) with zeros
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define block dimensions: each block covers a tile of size (TILE_M, TILE_N)
    dim3 block(TILE_M, TILE_N);
    // Grid dimensions: x and y cover the output matrix, z partitions the K dimension into chunks
    dim3 grid((M + TILE_M - 1) / TILE_M,
              (N + TILE_N - 1) / TILE_N,
              (K + BLOCK_K - 1) / BLOCK_K);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the kernel
    tiledAtomicKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using tiled reduction with atomic adds (CUDA)");
}
