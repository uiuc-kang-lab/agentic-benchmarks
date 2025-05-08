#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define THREAD_TILE 2
#define MAX_MATRIX_DIM 8192

// Constant memory for matrix dimension and number of tiles
__constant__ int d_N;
__constant__ int d_num_tiles;

// Each block computes a BLOCK_SIZE x BLOCK_SIZE tile of the output matrix. 
// Each thread in a block computes a 2x2 sub-tile, ensuring an even workload distribution.
__global__ void matmul_kernel_even(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C) {
    // Calculate the starting row and column for this block
    int blockRow = blockIdx.y * BLOCK_SIZE;
    int blockCol = blockIdx.x * BLOCK_SIZE;

    // Thread indices within the block (each thread computes a 2x2 tile)
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    // Global indices for the top-left element of this thread's 2x2 sub-tile
    int row = blockRow + ty * THREAD_TILE;
    int col = blockCol + tx * THREAD_TILE;

    // Registers to accumulate the 2x2 sub-tile results
    float regC00 = 0.0f, regC01 = 0.0f, regC10 = 0.0f, regC11 = 0.0f;

    // Shared memory to store a BLOCK_SIZE x BLOCK_SIZE tile from A and B
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over tiles along the inner dimension
    for (int t = 0; t < d_num_tiles; t++) {
        // Compute global indices for loading a 2x2 sub-tile from A
        int aRow0 = row;
        int aRow1 = row + 1;
        int aCol0 = t * BLOCK_SIZE + tx * THREAD_TILE;
        int aCol1 = aCol0 + 1;

        // Each thread loads a 2x2 block from A into shared memory, if within bounds
        s_A[ty * THREAD_TILE + 0][tx * THREAD_TILE + 0] = (aRow0 < d_N && aCol0 < d_N) ? A[aRow0 * d_N + aCol0] : 0.0f;
        s_A[ty * THREAD_TILE + 0][tx * THREAD_TILE + 1] = (aRow0 < d_N && aCol1 < d_N) ? A[aRow0 * d_N + aCol1] : 0.0f;
        s_A[ty * THREAD_TILE + 1][tx * THREAD_TILE + 0] = (aRow1 < d_N && aCol0 < d_N) ? A[aRow1 * d_N + aCol0] : 0.0f;
        s_A[ty * THREAD_TILE + 1][tx * THREAD_TILE + 1] = (aRow1 < d_N && aCol1 < d_N) ? A[aRow1 * d_N + aCol1] : 0.0f;

        // Compute global indices for loading a 2x2 sub-tile from B
        int bRow0 = t * BLOCK_SIZE + ty * THREAD_TILE;
        int bRow1 = bRow0 + 1;
        int bCol0 = col;
        int bCol1 = col + 1;

        s_B[ty * THREAD_TILE + 0][tx * THREAD_TILE + 0] = (bRow0 < d_N && bCol0 < d_N) ? B[bRow0 * d_N + bCol0] : 0.0f;
        s_B[ty * THREAD_TILE + 0][tx * THREAD_TILE + 1] = (bRow0 < d_N && bCol1 < d_N) ? B[bRow0 * d_N + bCol1] : 0.0f;
        s_B[ty * THREAD_TILE + 1][tx * THREAD_TILE + 0] = (bRow1 < d_N && bCol0 < d_N) ? B[bRow1 * d_N + bCol0] : 0.0f;
        s_B[ty * THREAD_TILE + 1][tx * THREAD_TILE + 1] = (bRow1 < d_N && bCol1 < d_N) ? B[bRow1 * d_N + bCol1] : 0.0f;

        __syncthreads();

        // Compute the dot product for the 2x2 sub-tile using the loaded shared memory tiles
        for (int k = 0; k < BLOCK_SIZE; k++) {
            float a0 = s_A[ty * THREAD_TILE + 0][k];
            float a1 = s_A[ty * THREAD_TILE + 1][k];
            float b0 = s_B[k][tx * THREAD_TILE + 0];
            float b1 = s_B[k][tx * THREAD_TILE + 1];
            regC00 += a0 * b0;
            regC01 += a0 * b1;
            regC10 += a1 * b0;
            regC11 += a1 * b1;
        }

        __syncthreads();
    }

    // Write the 2x2 computed tile back to global memory
    if (row < d_N && col < d_N)
        C[row * d_N + col] = regC00;
    if (row < d_N && (col + 1) < d_N)
        C[row * d_N + (col + 1)] = regC01;
    if ((row + 1) < d_N && col < d_N)
        C[(row + 1) * d_N + col] = regC10;
    if ((row + 1) < d_N && (col + 1) < d_N)
        C[(row + 1) * d_N + (col + 1)] = regC11;
}

// C++ interface (Pybind11 binding)
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same dimensions");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Copy matrix dimension and number of tiles to constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_num_tiles, &num_tiles, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Define the thread block dimensions: each block has (BLOCK_SIZE/THREAD_TILE) x (BLOCK_SIZE/THREAD_TILE) threads
    dim3 threads(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel_even<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Even Distribution Matrix Multiplication (CUDA)");
}
