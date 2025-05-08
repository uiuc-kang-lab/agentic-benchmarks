#include <torch/extension.h>
#include <cuda_runtime.h>

// Constants for tiling dimensions
#define TILE_DIM 32
#define BLOCK_SIZE 16
#define THREAD_TILE 2

// Optimized device function to load tiles from A into shared memory
__device__ void loadTileA(const float* __restrict__ A, float As[TILE_DIM][TILE_DIM], int M, int K, int tileIdx, int threadId, int numThreads, int blockRow) {
    for (int i = threadId; i < TILE_DIM * TILE_DIM; i += numThreads) {
        int localRow = i / TILE_DIM;
        int localCol = i % TILE_DIM;
        int globalRow = blockRow + localRow;
        int globalCol = tileIdx * TILE_DIM + localCol;
        As[localRow][localCol] = (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
    }
}

// Optimized device function to load tiles from B into shared memory
__device__ void loadTileB(const float* __restrict__ B, float Bs[TILE_DIM][TILE_DIM], int K, int N, int tileIdx, int threadId, int numThreads, int blockCol) {
    for (int i = threadId; i < TILE_DIM * TILE_DIM; i += numThreads) {
        int localRow = i / TILE_DIM;
        int localCol = i % TILE_DIM;
        int globalRow = tileIdx * TILE_DIM + localRow;
        int globalCol = blockCol + localCol;
        Bs[localRow][localCol] = (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
    }
}

// Unified kernel combining efficient tiling & computation
__global__ void optimized_matmul_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int K, int N) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int blockRow = blockIdx.y * TILE_DIM;
    int blockCol = blockIdx.x * TILE_DIM;
    int row = blockRow + threadIdx.y * THREAD_TILE;
    int col = blockCol + threadIdx.x * THREAD_TILE;

    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = BLOCK_SIZE * BLOCK_SIZE;

    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
        loadTileA(A, As, M, K, tileIdx, threadId, numThreads, blockRow);
        loadTileB(B, Bs, K, N, tileIdx, threadId, numThreads, blockCol);
        __syncthreads();

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

    if (row < M && col < N)
        C[row * N + col] = c00;
    if (row < M && (col + 1) < N)
        C[row * N + col + 1] = c01;
    if ((row + 1) < M && col < N)
        C[(row + 1) * N + col] = c10;
    if ((row + 1) < M && (col + 1) < N)
        C[(row + 1) * N + col + 1] = c11;
}

// Host function
// Utilizing the optimized kernel for matrix multiplication
// A: [M x K], B: [K x N], C: [M x N]
torch::Tensor optimized_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    optimized_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_matmul_cuda, "Optimized matrix multiplication (CUDA)");
}
