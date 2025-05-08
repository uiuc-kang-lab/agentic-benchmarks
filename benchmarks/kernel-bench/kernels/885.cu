#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_SIZE 16
#define THREAD_TILE 2

// Device function to load a tile of A into shared memory
__device__ void loadTileA(const float* __restrict__ A, float As[TILE_DIM][TILE_DIM], int M, int K, int tileIdx, int threadId, int numThreads, int blockRow) {
    for (int i = threadId; i < TILE_DIM * TILE_DIM; i += numThreads) {
        int localRow = i / TILE_DIM;
        int localCol = i % TILE_DIM;
        int globalRow = blockRow + localRow;
        int globalCol = tileIdx * TILE_DIM + localCol;
        if (globalRow < M && globalCol < K)
            As[localRow][localCol] = A[globalRow * K + globalCol];
        else
            As[localRow][localCol] = 0.0f;
    }
}

// Device function to load a tile of B into shared memory
__device__ void loadTileB(const float* __restrict__ B, float Bs[TILE_DIM][TILE_DIM], int K, int N, int tileIdx, int threadId, int numThreads, int blockCol) {
    for (int i = threadId; i < TILE_DIM * TILE_DIM; i += numThreads) {
        int localRow = i / TILE_DIM;
        int localCol = i % TILE_DIM;
        int globalRow = tileIdx * TILE_DIM + localRow;
        int globalCol = blockCol + localCol;
        if (globalRow < K && globalCol < N)
            Bs[localRow][localCol] = B[globalRow * N + globalCol];
        else
            Bs[localRow][localCol] = 0.0f;
    }
}

// Device function to compute a 2x2 sub-tile
__device__ void computeSubTile(float& c00, float& c01, float& c10, float& c11, float As[TILE_DIM][TILE_DIM], float Bs[TILE_DIM][TILE_DIM], int k, int ty, int tx) {
    float a0 = As[ty * THREAD_TILE + 0][k];
    float a1 = As[ty * THREAD_TILE + 1][k];
    float b0 = Bs[k][tx * THREAD_TILE + 0];
    float b1 = Bs[k][tx * THREAD_TILE + 1];
    c00 += a0 * b0;
    c01 += a0 * b1;
    c10 += a1 * b0;
    c11 += a1 * b1;
}

// Kernel function
__global__ void matmul_kernel(const float* __restrict__ A,
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
            computeSubTile(c00, c01, c10, c11, As, Bs, k, threadIdx.y, threadIdx.x);
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
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication (CUDA) with modular device functions");
}