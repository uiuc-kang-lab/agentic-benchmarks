#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Define block dimensions
#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)

// This kernel uses double buffering in shared memory to reduce global memory latency.
// Two shared memory buffers are used to preload the next tile while computing the current one.
// Proper __syncthreads() calls ensure data consistency and avoid race conditions.
__global__ void double_buffered_matmul_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int M, int K, int N) {
    // Compute the starting indices of the output tile for this block
    int blockRow = blockIdx.y * TILE_DIM;
    int blockCol = blockIdx.x * TILE_DIM;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Allocate double buffers in shared memory for tiles of A and B
    __shared__ float As[2][TILE_DIM][TILE_DIM];
    __shared__ float Bs[2][TILE_DIM][TILE_DIM];

    // Calculate the number of tiles needed along the K dimension
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    int currBuf = 0;  // current buffer index (0 or 1)

    // Preload the first tile (tile 0) into the current buffer
    {
        int t = 0;
        int aTileColBase = t * TILE_DIM;
        int bTileRowBase = t * TILE_DIM;
        // Each thread loads a 2x2 sub-block for matrix A
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int aRow = blockRow + ty + i * BLOCK_SIZE;
                int aCol = aTileColBase + tx + j * BLOCK_SIZE;
                if (aRow < M && aCol < K)
                    As[currBuf][ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = A[aRow * K + aCol];
                else
                    As[currBuf][ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = 0.0f;
            }
        }
        // Each thread loads a 2x2 sub-block for matrix B
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int bRow = bTileRowBase + ty + i * BLOCK_SIZE;
                int bCol = blockCol + tx + j * BLOCK_SIZE;
                if (bRow < K && bCol < N)
                    Bs[currBuf][ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = B[bRow * N + bCol];
                else
                    Bs[currBuf][ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = 0.0f;
            }
        }
    }
    __syncthreads();

    // Initialize accumulators for the 2x2 output submatrix computed by each thread
    float Cvalue00 = 0.0f, Cvalue01 = 0.0f, Cvalue10 = 0.0f, Cvalue11 = 0.0f;

    // Loop over all tiles along the K dimension
    for (int t = 0; t < numTiles; t++) {
        int nextBuf = currBuf ^ 1;  // alternate buffer index
        // Preload the next tile into the alternate buffer if available
        if (t < numTiles - 1) {
            int aTileColBase = (t + 1) * TILE_DIM;
            int bTileRowBase = (t + 1) * TILE_DIM;
            // Load tile for A into buffer 'nextBuf'
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    int aRow = blockRow + ty + i * BLOCK_SIZE;
                    int aCol = aTileColBase + tx + j * BLOCK_SIZE;
                    if (aRow < M && aCol < K)
                        As[nextBuf][ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = A[aRow * K + aCol];
                    else
                        As[nextBuf][ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = 0.0f;
                }
            }
            // Load tile for B into buffer 'nextBuf'
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    int bRow = bTileRowBase + ty + i * BLOCK_SIZE;
                    int bCol = blockCol + tx + j * BLOCK_SIZE;
                    if (bRow < K && bCol < N)
                        Bs[nextBuf][ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = B[bRow * N + bCol];
                    else
                        Bs[nextBuf][ty + i * BLOCK_SIZE][tx + j * BLOCK_SIZE] = 0.0f;
                }
            }
        }

        // Synchronize to ensure the current tile data is ready for computation
        __syncthreads();

        // Compute the partial results for the current tile using data in the current buffer
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_val0 = As[currBuf][ty][k];
            float a_val1 = As[currBuf][ty + BLOCK_SIZE][k];
            float b_val0 = Bs[currBuf][k][tx];
            float b_val1 = Bs[currBuf][k][tx + BLOCK_SIZE];
            Cvalue00 += a_val0 * b_val0;
            Cvalue01 += a_val0 * b_val1;
            Cvalue10 += a_val1 * b_val0;
            Cvalue11 += a_val1 * b_val1;
        }

        // Swap the buffers for the next iteration
        currBuf ^= 1;
        __syncthreads();
    }

    // Write the computed 2x2 submatrix results from registers to global memory
    int row0 = blockRow + ty;
    int col0 = blockCol + tx;
    int row1 = row0 + BLOCK_SIZE;
    int col1 = col0 + BLOCK_SIZE;

    if (row0 < M && col0 < N) C[row0 * N + col0] = Cvalue00;
    if (row0 < M && col1 < N) C[row0 * N + col1] = Cvalue01;
    if (row1 < M && col0 < N) C[row1 * N + col0] = Cvalue10;
    if (row1 < M && col1 < N) C[row1 * N + col1] = Cvalue11;
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

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
    double_buffered_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Double-buffered matrix multiplication (CUDA)");
}
