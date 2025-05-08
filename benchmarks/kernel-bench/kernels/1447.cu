#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile sizes for shared memory tiling and thread coarsening
#define TILE_SIZE 32   // Each tile covers a 32x32 submatrix
#define BLOCK_SIZE 16  // Thread block dimensions (16x16 threads) with each thread computing a 2x2 block


// Optimized kernel using thread coarsening: each thread computes a 2x2 block, ensuring even workload distribution.
__global__ void matmul_kernel_coarsened(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int N) {
    // Calculate the starting indices of the output tile for this block
    int blockRow = blockIdx.y * TILE_SIZE;
    int blockCol = blockIdx.x * TILE_SIZE;

    // Each thread block is of size BLOCK_SIZE x BLOCK_SIZE and each thread computes a 2x2 sub-block
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Global output coordinate for the top-left element computed by this thread
    int row = blockRow + 2 * ty;
    int col = blockCol + 2 * tx;

    // Register accumulators for the 2x2 output computed by this thread
    float sum00 = 0.0f, sum01 = 0.0f, sum10 = 0.0f, sum11 = 0.0f;

    // Shared memory to hold a tile of A and B
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    // Number of tiles to iterate over in the k-dimension
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    // Loop over all tiles in the k-dimension
    for (int t = 0; t < numTiles; t++) {
        // Compute global indices for the A tile to be loaded
        int A_tile_row = blockRow + 2 * ty;       // Starting row for the 2x2 block from A
        int A_tile_col = t * TILE_SIZE + 2 * tx;      // Starting column within the tile
        
        // Each thread loads a 2x2 block from A into shared memory
        // First row of the 2x2 block
        if (A_tile_row < N) {
            if (A_tile_col < N)
                s_A[2 * ty][2 * tx] = A[A_tile_row * N + A_tile_col];
            else
                s_A[2 * ty][2 * tx] = 0.0f;
            if ((A_tile_col + 1) < N)
                s_A[2 * ty][2 * tx + 1] = A[A_tile_row * N + A_tile_col + 1];
            else
                s_A[2 * ty][2 * tx + 1] = 0.0f;
        } else {
            s_A[2 * ty][2 * tx] = 0.0f;
            s_A[2 * ty][2 * tx + 1] = 0.0f;
        }

        // Second row of the 2x2 block from A
        if ((A_tile_row + 1) < N) {
            if (A_tile_col < N)
                s_A[2 * ty + 1][2 * tx] = A[(A_tile_row + 1) * N + A_tile_col];
            else
                s_A[2 * ty + 1][2 * tx] = 0.0f;
            if ((A_tile_col + 1) < N)
                s_A[2 * ty + 1][2 * tx + 1] = A[(A_tile_row + 1) * N + A_tile_col + 1];
            else
                s_A[2 * ty + 1][2 * tx + 1] = 0.0f;
        } else {
            s_A[2 * ty + 1][2 * tx] = 0.0f;
            s_A[2 * ty + 1][2 * tx + 1] = 0.0f;
        }

        // Compute global indices for the B tile to be loaded
        int B_tile_row = t * TILE_SIZE + 2 * ty;
        int B_tile_col = blockCol + 2 * tx;
        
        // Each thread loads a 2x2 block from B into shared memory
        // First row of the 2x2 block from B
        if (B_tile_row < N) {
            if (B_tile_col < N)
                s_B[2 * ty][2 * tx] = B[B_tile_row * N + B_tile_col];
            else
                s_B[2 * ty][2 * tx] = 0.0f;
            if ((B_tile_col + 1) < N)
                s_B[2 * ty][2 * tx + 1] = B[B_tile_row * N + B_tile_col + 1];
            else
                s_B[2 * ty][2 * tx + 1] = 0.0f;
        } else {
            s_B[2 * ty][2 * tx] = 0.0f;
            s_B[2 * ty][2 * tx + 1] = 0.0f;
        }
        
        // Second row of the 2x2 block from B
        if ((B_tile_row + 1) < N) {
            if (B_tile_col < N)
                s_B[2 * ty + 1][2 * tx] = B[(B_tile_row + 1) * N + B_tile_col];
            else
                s_B[2 * ty + 1][2 * tx] = 0.0f;
            if ((B_tile_col + 1) < N)
                s_B[2 * ty + 1][2 * tx + 1] = B[(B_tile_row + 1) * N + B_tile_col + 1];
            else
                s_B[2 * ty + 1][2 * tx + 1] = 0.0f;
        } else {
            s_B[2 * ty + 1][2 * tx] = 0.0f;
            s_B[2 * ty + 1][2 * tx + 1] = 0.0f;
        }

        __syncthreads();

        // Compute the partial results for the 2x2 output block, iterating over the loaded tile
        for (int k = 0; k < TILE_SIZE; k++) {
            float a0 = s_A[2 * ty][k];
            float a1 = s_A[2 * ty + 1][k];
            float b0 = s_B[k][2 * tx];
            float b1 = s_B[k][2 * tx + 1];
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }

        __syncthreads();
    }

    // Write the 2x2 block computed by this thread into the output matrix C, with boundary checks
    if (row < N && col < N)
        C[row * N + col] = sum00;
    if (row < N && (col + 1) < N)
        C[row * N + col + 1] = sum01;
    if ((row + 1) < N && col < N)
        C[(row + 1) * N + col] = sum10;
    if ((row + 1) < N && (col + 1) < N)
        C[(row + 1) * N + col + 1] = sum11;
}

// C++ interface exposed via Pybind11
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int N = A.size(0);
    
    // Configure grid: each block computes a TILE_SIZE x TILE_SIZE output tile
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    // Each block uses BLOCK_SIZE x BLOCK_SIZE threads (each thread computes a 2x2 output block)
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    matmul_kernel_coarsened<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized MatMul with Thread Coarsening (CUDA)");
}
