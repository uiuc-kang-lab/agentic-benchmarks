#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_WIDTH 32

// Kernel: each thread computes a 2x2 sub-block of the output matrix C using register tiling.
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int M, int K, int N) {
    // Allocate shared memory for a tile of A and B
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    // Each block computes a TILE_WIDTH x TILE_WIDTH tile of C.
    // We use a 2x2 register block per thread. Thus, block dimensions are (TILE_WIDTH/2, TILE_WIDTH/2).
    int tx = threadIdx.x; // Range: 0 to (TILE_WIDTH/2 - 1)
    int ty = threadIdx.y; // Range: 0 to (TILE_WIDTH/2 - 1)
    
    // Compute the starting row and col for this thread's 2x2 sub-block
    int row = blockIdx.y * TILE_WIDTH + ty * 2;
    int col = blockIdx.x * TILE_WIDTH + tx * 2;
    // Precompute boundary validity and row base indices to reduce redundant computations
    const bool validRow0 = (row < M);
    const bool validRow1 = (row + 1 < M);
    const bool validCol0 = (col < N);
    const bool validCol1 = (col + 1 < N);
    const int aBase0 = row * K;
    const int aBase1 = validRow1 ? (row + 1) * K : 0;

    // Registers for the 2x2 sub-block accumulation
    float Cvalue00 = 0.0f;
    float Cvalue01 = 0.0f;
    float Cvalue10 = 0.0f;
    float Cvalue11 = 0.0f;

    // Loop over tiles in the K dimension
    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; t++) {
        // Each thread loads 4 elements from A into shared memory for its corresponding rows.
        int aCol0 = t * TILE_WIDTH + 2 * tx;
        int aCol1 = t * TILE_WIDTH + 2 * tx + 1;

        if (row < M && aCol0 < K)
            As[ty * 2][2 * tx] = A[row * K + aCol0];
        else
            As[ty * 2][2 * tx] = 0.0f;

        if (row < M && aCol1 < K)
            As[ty * 2][2 * tx + 1] = A[row * K + aCol1];
        else
            As[ty * 2][2 * tx + 1] = 0.0f;

        if ((row + 1) < M && aCol0 < K)
            As[ty * 2 + 1][2 * tx] = A[(row + 1) * K + aCol0];
        else
            As[ty * 2 + 1][2 * tx] = 0.0f;

        if ((row + 1) < M && aCol1 < K)
            As[ty * 2 + 1][2 * tx + 1] = A[(row + 1) * K + aCol1];
        else
            As[ty * 2 + 1][2 * tx + 1] = 0.0f;

        // Each thread loads 4 elements from B into shared memory for its corresponding columns.
        int bRow0 = t * TILE_WIDTH + 2 * ty;
        int bRow1 = t * TILE_WIDTH + 2 * ty + 1;
        int bCol0 = col;
        int bCol1 = col + 1;

        if (bRow0 < K && bCol0 < N)
            Bs[2 * ty][2 * tx] = B[bRow0 * N + bCol0];
        else
            Bs[2 * ty][2 * tx] = 0.0f;

        if (bRow0 < K && bCol1 < N)
            Bs[2 * ty][2 * tx + 1] = B[bRow0 * N + bCol1];
        else
            Bs[2 * ty][2 * tx + 1] = 0.0f;

        if (bRow1 < K && bCol0 < N)
            Bs[2 * ty + 1][2 * tx] = B[bRow1 * N + bCol0];
        else
            Bs[2 * ty + 1][2 * tx] = 0.0f;

        if (bRow1 < K && bCol1 < N)
            Bs[2 * ty + 1][2 * tx + 1] = B[bRow1 * N + bCol1];
        else
            Bs[2 * ty + 1][2 * tx + 1] = 0.0f;

        __syncthreads();

        // Compute the partial 2x2 product for the current tile
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k++) {
            float a0 = As[ty * 2][k];
            float a1 = As[ty * 2 + 1][k];
            float b0 = Bs[k][2 * tx];
            float b1 = Bs[k][2 * tx + 1];
            Cvalue00 += a0 * b0;
            Cvalue01 += a0 * b1;
            Cvalue10 += a1 * b0;
            Cvalue11 += a1 * b1;
        }

        __syncthreads();
    }

    // Write the computed 2x2 block back to global memory
    if (row < M && col < N)
        C[row * N + col] = Cvalue00;
    if (row < M && (col + 1) < N)
        C[row * N + (col + 1)] = Cvalue01;
    if ((row + 1) < M && col < N)
        C[(row + 1) * N + col] = Cvalue10;
    if ((row + 1) < M && (col + 1) < N)
        C[(row + 1) * N + (col + 1)] = Cvalue11;
}


torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    // Configure block and grid dimensions.
    // Each block computes a TILE_WIDTH x TILE_WIDTH tile of C using (TILE_WIDTH/2 x TILE_WIDTH/2) threads.
    dim3 blockDim(TILE_WIDTH / 2, TILE_WIDTH / 2);  // e.g., 16x16 threads
    dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with register tiling (CUDA)");
}
