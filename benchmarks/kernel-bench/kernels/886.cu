#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile dimensions for shared memory tiling
#define TILE_DIM 32
// Block dimensions for launching kernel (each block has 16x16 threads)
#define BLOCK_SIZE 16
// Each thread computes a 2x2 sub-tile
#define THREAD_TILE 2

// Warp size
#define WARP_SIZE 32

// Optimized CUDA kernel using warp-level primitives for reduction
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int K, int N) {
    // Each block computes a TILE_DIM x TILE_DIM sub-matrix of C
    // Block starting indices in C
    int blockRow = blockIdx.y * TILE_DIM;
    int blockCol = blockIdx.x * TILE_DIM;

    // Each thread computes a 2x2 sub-tile within the block
    int row = blockRow + threadIdx.y * THREAD_TILE;
    int col = blockCol + threadIdx.x * THREAD_TILE;

    // Accumulators for the 2x2 sub-tile
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    // Allocate shared memory for a tile of A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
        // Use all threads in the block to load the TILE_DIM x TILE_DIM tile from global memory
        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        int numThreads = BLOCK_SIZE * BLOCK_SIZE;

        // Load tile from A into shared memory
        for (int i = threadId; i < TILE_DIM * TILE_DIM; i += numThreads) {
            int localRow = i / TILE_DIM;
            int localCol = i % TILE_DIM;
            int globalRow = blockIdx.y * TILE_DIM + localRow;
            int globalCol = tileIdx * TILE_DIM + localCol;
            if (globalRow < M && globalCol < K)
                As[localRow][localCol] = A[globalRow * K + globalCol];
            else
                As[localRow][localCol] = 0.0f;
        }

        // Load tile from B into shared memory
        for (int i = threadId; i < TILE_DIM * TILE_DIM; i += numThreads) {
            int localRow = i / TILE_DIM;
            int localCol = i % TILE_DIM;
            int globalRow = tileIdx * TILE_DIM + localRow;
            int globalCol = blockIdx.x * TILE_DIM + localCol;
            if (globalRow < K && globalCol < N)
                Bs[localRow][localCol] = B[globalRow * N + globalCol];
            else
                Bs[localRow][localCol] = 0.0f;
        }

        __syncthreads();

        // Compute partial products for the 2x2 sub-tile computed by this thread
        for (int k = 0; k < TILE_DIM; k++) {
            float a0 = As[threadIdx.y * THREAD_TILE + 0][k];
            float a1 = As[threadIdx.y * THREAD_TILE + 1][k];
            float b0 = Bs[k][threadIdx.x * THREAD_TILE + 0];
            float b1 = Bs[k][threadIdx.x * THREAD_TILE + 1];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    // Write the 2x2 sub-tile result back to global memory
    if (row < M && col < N) {
        C[row * N + col] = c00;
        if (col + 1 < N)
            C[row * N + col + 1] = c01;
        if (row + 1 < M) {
            C[(row + 1) * N + col] = c10;
            if (col + 1 < N)
                C[(row + 1) * N + col + 1] = c11;
        }
    }
}

// Host function that wraps the CUDA kernel launch
// A: [M x K], B: [K x N], C: [M x N]
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Calculate grid dimensions based on TILE_DIM
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch the kernel
    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication (CUDA) with warp-level reduction");
}