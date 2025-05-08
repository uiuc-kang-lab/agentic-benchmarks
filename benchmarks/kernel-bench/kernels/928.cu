#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 32

// Optimized tiled matrix multiplication kernel that minimizes __syncthreads() calls.
// It only synchronizes when necessary: after loading into shared memory and before starting a new tile iteration.
__global__ void matmul_min_sync_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Load tile from A into shared memory
        int a_col = t * TILE_SIZE + tx;
        if (row < M && a_col < K) {
            As[ty][tx] = A[row * K + a_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        // Load tile from B into shared memory
        int b_row = t * TILE_SIZE + ty;
        if (col < N && b_row < K) {
            Bs[ty][tx] = B[b_row * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        // Ensure the tile is loaded before computing
        if (t < numTiles - 1) __syncthreads();

        // Compute the dot product for the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        // Synchronize only if there is another tile to process
        if (t < numTiles - 1) {
            if (t < numTiles - 1) __syncthreads();
        }
    }

    // Write the computed value to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function called from PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_min_sync_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with minimized synchronizations (CUDA)");
}
