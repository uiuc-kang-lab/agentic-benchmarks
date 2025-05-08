#include <torch/extension.h>
#include <cuda_runtime.h>

// Define tile size for shared memory and output tiles
#define TILE_SIZE 16

// CUDA kernel that uses grid-stride loops over output tiles
// Each block computes one TILE_SIZE x TILE_SIZE tile of C, and if the matrix is larger
// than the number of blocks, the block iterates (strides) over multiple tiles.
// Within each tile, standard tiling over the K-dimension with shared memory is used.

__global__ void matmul_stride_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
    // Shared memory for a tile of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Compute the total number of tiles in the row and col dimensions for C
    int numTilesRow = (M + TILE_SIZE - 1) / TILE_SIZE;
    int numTilesCol = (N + TILE_SIZE - 1) / TILE_SIZE;
    int numTilesK   = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over output tile indices using grid-stride loops
    for (int tileRow = blockIdx.y; tileRow < numTilesRow; tileRow += gridDim.y) {
        for (int tileCol = blockIdx.x; tileCol < numTilesCol; tileCol += gridDim.x) {
            float sum = 0.0f;
            // Compute the global row and col for this tile element
            int row = tileRow * TILE_SIZE + threadIdx.y;
            int col = tileCol * TILE_SIZE + threadIdx.x;

            // Loop over tiles in the K dimension
            for (int t = 0; t < numTilesK; t++) {
                // Compute column index for A and row index for B
                int a_col = t * TILE_SIZE + threadIdx.x;
                int b_row = t * TILE_SIZE + threadIdx.y;

                // Load element from A into shared memory
                if (row < M && a_col < K)
                    As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
                else
                    As[threadIdx.y][threadIdx.x] = 0.0f;

                // Load element from B into shared memory
                if (b_row < K && col < N)
                    Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
                else
                    Bs[threadIdx.y][threadIdx.x] = 0.0f;

                // __syncthreads(); // Commented out to allow for overlapping computation and memory transfers.

                // Perform the dot product for the current tile
                #pragma unroll
                for (int k = 0; k < TILE_SIZE; k++) {
                    sum = __fmaf_rn(As[threadIdx.y][k], Bs[k][threadIdx.x], sum);
                }

                // __syncthreads(); // Commented out to allow for overlapping computation and memory transfers.
            }

            // Write the computed value to C if within bounds
            if (row < M && col < N) {
                C[row * N + col] = sum;
            }
        }
    }
}

// The forward function is exposed to PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Compute number of tiles in each dimension
    int numTilesRow = (M + TILE_SIZE - 1) / TILE_SIZE;
    int numTilesCol = (N + TILE_SIZE - 1) / TILE_SIZE;

    // To demonstrate stride looping, we limit grid dimensions to a fixed maximum (e.g., 32 per dimension)
    int gridY = (numTilesRow < 32) ? numTilesRow : 32;
    int gridX = (numTilesCol < 32) ? numTilesCol : 32;

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(gridX, gridY);

    matmul_stride_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with stride loops (CUDA)");
}
