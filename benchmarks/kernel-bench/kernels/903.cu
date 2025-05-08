#include <torch/extension.h>
#include <cuda_runtime.h>

// Tiling parameters
#define TILE_DIM 32
#define BLOCK_SIZE 16
#define THREAD_TILE 2

// Custom matrix multiplication kernel using double buffering with reduced __syncthreads()
__global__ void custom_matmul_kernel_db(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int K, int N) {
    // Double buffered shared memory for A and B tiles
    __shared__ float As[2][TILE_DIM][TILE_DIM];
    __shared__ float Bs[2][TILE_DIM][TILE_DIM];

    // Calculate block indices
    int blockRow = blockIdx.y * TILE_DIM;
    int blockCol = blockIdx.x * TILE_DIM;

    // Each thread computes a 2x2 sub-tile in C
    int row = blockRow + threadIdx.y * THREAD_TILE;
    int col = blockCol + threadIdx.x * THREAD_TILE;

    // Flattened thread index for loading data
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = BLOCK_SIZE * BLOCK_SIZE;

    // Accumulators for the 2x2 sub-tile
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    // Buffer indices for double buffering
    int curr = 0, next = 1;

    // Preload the first tile (tile index 0) into buffer 'curr' with coalesced access
    for (int i = 0; i < TILE_DIM; i += BLOCK_SIZE) {
        int localRow = threadIdx.y + i;
        int localCol = threadIdx.x;
        int globalRow = blockRow + localRow;
        int globalCol = localCol;
        if (globalRow < M && globalCol < K)
            As[curr][localRow][localCol] = A[globalRow * K + globalCol];
        else
            As[curr][localRow][localCol] = 0.0f;
    }
    for (int i = 0; i < TILE_DIM; i += BLOCK_SIZE) {
        int localRow = threadIdx.y + i;
        int localCol = threadIdx.x;
        int globalRow = localRow;
        int globalCol = blockCol + localCol;
        if (globalRow < K && globalCol < N)
            Bs[curr][localRow][localCol] = B[globalRow * N + globalCol];
        else
            Bs[curr][localRow][localCol] = 0.0f;
    }
    __syncthreads();

    // Loop over all tiles except the last one
    for (int tileIdx = 0; tileIdx < numTiles - 1; tileIdx++) {
        // Compute partial results using the current tile in buffer 'curr'
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            int aRow = threadIdx.y * THREAD_TILE; // starting row in the tile for A
            int bCol = threadIdx.x * THREAD_TILE; // starting col in the tile for B
            float a0 = As[curr][aRow + 0][k];
            float a1 = As[curr][aRow + 1][k];
            float b0 = Bs[curr][k][bCol + 0];
            float b1 = Bs[curr][k][bCol + 1];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        // Load the next tile (tileIdx+1) into buffer 'next'
        int nextTile = tileIdx + 1;
        for (int i = threadId; i < TILE_DIM * TILE_DIM; i += numThreads) {
            int localRow = i / TILE_DIM;
            int localCol = i % TILE_DIM;
            int globalRow = blockRow + localRow;
            int globalCol = nextTile * TILE_DIM + localCol;
            if (globalRow < M && globalCol < K)
                As[next][localRow][localCol] = A[globalRow * K + globalCol];
            else
                As[next][localRow][localCol] = 0.0f;
        }
        for (int i = threadId; i < TILE_DIM * TILE_DIM; i += numThreads) {
            int localRow = i / TILE_DIM;
            int localCol = i % TILE_DIM;
            int globalRow = nextTile * TILE_DIM + localRow;
            int globalCol = blockCol + localCol;
            if (globalRow < K && globalCol < N)
                Bs[next][localRow][localCol] = B[globalRow * N + globalCol];
            else
                Bs[next][localRow][localCol] = 0.0f;
        }

        // Synchronize to ensure the next tile is fully loaded
        __syncthreads();

        // Swap the buffers
        int temp = curr;
        curr = next;
        next = temp;
    }

    // Process the last tile from buffer 'curr'
    #pragma unroll
    for (int k = 0; k < TILE_DIM; k++) {
        int aRow = threadIdx.y * THREAD_TILE;
        int bCol = threadIdx.x * THREAD_TILE;
        float a0 = As[curr][aRow + 0][k];
        float a1 = As[curr][aRow + 1][k];
        float b0 = Bs[curr][k][bCol + 0];
        float b1 = Bs[curr][k][bCol + 1];
        c00 += a0 * b0;
        c01 += a0 * b1;
        c10 += a1 * b0;
        c11 += a1 * b1;
    }

    // Write the 2x2 computed sub-tile to global memory
    if (row < M && col < N)
        C[row * N + col] = c00;
    if (row < M && (col + 1) < N)
        C[row * N + col + 1] = c01;
    if ((row + 1) < M && col < N)
        C[(row + 1) * N + col] = c10;
    if ((row + 1) < M && (col + 1) < N)
        C[(row + 1) * N + col + 1] = c11;
}

// Host function to launch the custom kernel
// A: [M x K], B: [K x N], C: [M x N]

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    custom_matmul_kernel_db<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication (CUDA) with double buffering and reduced synchronizations");
}
