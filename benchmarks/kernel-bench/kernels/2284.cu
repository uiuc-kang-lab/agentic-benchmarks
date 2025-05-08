#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Set the tile size (assumed to be 16 for manual unrolling)
#define TILE_SIZE 16

// CUDA kernel to compute C = A.T * B using shared memory tiling with manual loop unrolling in the inner loop.
// A: shape (K, M), B: shape (K, N), C: shape (M, N)
// Note: A is stored as (K, M) so that A.T(i,k) = A(k,i).
__global__ void matMulManualUnrollKernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int K, int M, int N) {
    // Calculate the row index of C (and column index of A)
    int row = blockIdx.x * TILE_SIZE + threadIdx.y;
    // Calculate the column index of C (and B)
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;

    // Allocate shared memory for a tile of A (in transposed layout) and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Loop over tiles of the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load a tile of A (transposed) from global memory into shared memory
        int aIndex = t * TILE_SIZE + threadIdx.x;
        if (aIndex < K && row < M) {
            // A is originally (K, M): we access A(aIndex, row) which is A[aIndex * M + row]
            tileA[threadIdx.y][threadIdx.x] = A[aIndex * M + row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of B from global memory into shared memory
        int bIndex = t * TILE_SIZE + threadIdx.y;
        if (bIndex < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bIndex * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // __syncthreads();

        // Manually unroll the inner loop that computes the dot-product for the tile.
        // Unrolling 16 iterations since TILE_SIZE is 16.
        float a0 = tileA[threadIdx.y][0];
        float a1 = tileA[threadIdx.y][1];
        float a2 = tileA[threadIdx.y][2];
        float a3 = tileA[threadIdx.y][3];
        float a4 = tileA[threadIdx.y][4];
        float a5 = tileA[threadIdx.y][5];
        float a6 = tileA[threadIdx.y][6];
        float a7 = tileA[threadIdx.y][7];
        float a8 = tileA[threadIdx.y][8];
        float a9 = tileA[threadIdx.y][9];
        float a10 = tileA[threadIdx.y][10];
        float a11 = tileA[threadIdx.y][11];
        float a12 = tileA[threadIdx.y][12];
        float a13 = tileA[threadIdx.y][13];
        float a14 = tileA[threadIdx.y][14];
        float a15 = tileA[threadIdx.y][15];

        float b0 = tileB[0][threadIdx.x];
        float b1 = tileB[1][threadIdx.x];
        float b2 = tileB[2][threadIdx.x];
        float b3 = tileB[3][threadIdx.x];
        float b4 = tileB[4][threadIdx.x];
        float b5 = tileB[5][threadIdx.x];
        float b6 = tileB[6][threadIdx.x];
        float b7 = tileB[7][threadIdx.x];
        float b8 = tileB[8][threadIdx.x];
        float b9 = tileB[9][threadIdx.x];
        float b10 = tileB[10][threadIdx.x];
        float b11 = tileB[11][threadIdx.x];
        float b12 = tileB[12][threadIdx.x];
        float b13 = tileB[13][threadIdx.x];
        float b14 = tileB[14][threadIdx.x];
        float b15 = tileB[15][threadIdx.x];

        sum += a0*b0 + a1*b1 + a2*b2 + a3*b3 +
               a4*b4 + a5*b5 + a6*b6 + a7*b7 +
               a8*b8 + a9*b9 + a10*b10 + a11*b11 +
               a12*b12 + a13*b13 + a14*b14 + a15*b15;

        // __syncthreads();
    }

    // Write the computed sum to the output matrix C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function to be called from PyTorch via PyBind11
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // A: (K, M) and B: (K, N)
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate the output tensor C of shape (M, N)
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define block and grid sizes
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Get raw pointer to tensor data
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the CUDA kernel
    matMulManualUnrollKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B (CUDA) using manual loop unrolling in shared memory tiling");
}
