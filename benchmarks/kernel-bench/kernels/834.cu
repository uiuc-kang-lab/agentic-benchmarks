#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

// Define block and tile dimensions
// Each thread block has BLOCK_SIZE x BLOCK_SIZE threads
// Each thread computes a 2x2 submatrix, so the tile dimension is TILE_DIM = BLOCK_SIZE * 2
#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)

// Kernel: Each thread computes a 2x2 block of C using register tiling.
// This distributes work evenly and increases the arithmetic intensity per thread.
__global__ void reg_tile_matmul_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int K, int N) {
    // Compute the starting global indices for the 2x2 submatrix computed by this thread
    int row0 = blockIdx.y * TILE_DIM + threadIdx.y;
    int col0 = blockIdx.x * TILE_DIM + threadIdx.x;
    int row1 = row0 + BLOCK_SIZE;   // second row computed by this thread
    int col1 = col0 + BLOCK_SIZE;   // second column computed by this thread

    // Accumulators for the 2x2 submatrix
    float Cvalue00 = 0.0f, Cvalue01 = 0.0f, Cvalue10 = 0.0f, Cvalue11 = 0.0f;

    // Shared memory for tiles of A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Loop over the tiles of A and B along the K dimension
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int tile = 0; tile < numTiles; tile++) {
        int tStart = tile * TILE_DIM;

        // Load tile from A into shared memory
        // Each thread loads 4 elements from A, covering a 2x2 block in the shared tile
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int aRow = blockIdx.y * TILE_DIM + threadIdx.y + i * BLOCK_SIZE;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int aCol = tStart + threadIdx.x + j * BLOCK_SIZE;
                int sharedRow = threadIdx.y + i * BLOCK_SIZE;
                int sharedCol = threadIdx.x + j * BLOCK_SIZE;
                if (aRow < M && aCol < K) {
                    As[sharedRow][sharedCol] = A[aRow * K + aCol];
                } else {
                    As[sharedRow][sharedCol] = 0.0f;
                }
            }
        }

        // Load tile from B into shared memory
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int bRow = tStart + threadIdx.y + i * BLOCK_SIZE;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int bCol = blockIdx.x * TILE_DIM + threadIdx.x + j * BLOCK_SIZE;
                int sharedRow = threadIdx.y + i * BLOCK_SIZE;
                int sharedCol = threadIdx.x + j * BLOCK_SIZE;
                if (bRow < K && bCol < N) {
                    Bs[sharedRow][sharedCol] = B[bRow * N + bCol];
                } else {
                    Bs[sharedRow][sharedCol] = 0.0f;
                }
            }
        }

        __syncthreads();

        // Compute the partial results for the 2x2 submatrix
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_val0 = As[threadIdx.y][k];
            float a_val1 = As[threadIdx.y + BLOCK_SIZE][k];
            float b_val0 = Bs[k][threadIdx.x];
            float b_val1 = Bs[k][threadIdx.x + BLOCK_SIZE];
            Cvalue00 += a_val0 * b_val0;
            Cvalue01 += a_val0 * b_val1;
            Cvalue10 += a_val1 * b_val0;
            Cvalue11 += a_val1 * b_val1;
        }

        __syncthreads();
    }

    // Write the 2x2 block results to global memory (with boundary checks)
    if (row0 < M && col0 < N) {
        C[row0 * N + col0] = Cvalue00;
    }
    if (row0 < M && col1 < N) {
        C[row0 * N + col1] = Cvalue01;
    }
    if (row1 < M && col0 < N) {
        C[row1 * N + col0] = Cvalue10;
    }
    if (row1 < M && col1 < N) {
        C[row1 * N + col1] = Cvalue11;
    }
}

// Host function to launch the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Grid dimensions: each block computes a TILE_DIM x TILE_DIM output tile
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    reg_tile_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Register tiled matrix multiplication (CUDA)");
}
