#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define THREAD_TILE 2
#define UNROLL_FACTOR 8
#define MAX_MATRIX_DIM 8192

// Constant memory for matrix dimensions and number of tiles
__constant__ int d_N;
__constant__ int d_num_tiles;

// Combined kernel that uses aligned loads (__ldg) with manual loop unrolling
// This kernel loads 2x2 sub-tiles per thread and unrolls the inner loop to reduce overhead
__global__ void matmul_kernel_combined(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C) {
    // Shared memory tiles for A and B
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    // Determine global starting row and column for each thread's 2x2 sub-tile
    int row = blockIdx.y * BLOCK_SIZE + ty * THREAD_TILE;
    int col = blockIdx.x * BLOCK_SIZE + tx * THREAD_TILE;

    // 2x2 register tile to accumulate partial results
    float regC[THREAD_TILE][THREAD_TILE] = {0.0f};

    // Loop over all tiles in the k-dimension
    for (int t = 0; t < d_num_tiles; t++) {
        // Load sub-tile of A into shared memory using manual unrolling and __ldg
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                int aRow = row + i;
                int aCol = t * BLOCK_SIZE + tx * THREAD_TILE + j;
                s_A[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 
                    (aRow < d_N && aCol < d_N) ? __ldg(&A[aRow * d_N + aCol]) : 0.0f;
            }
        }

        // Load sub-tile of B into shared memory using manual unrolling and __ldg
        #pragma unroll
        for (int i = 0; i < THREAD_TILE; i++) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE; j++) {
                int bRow = t * BLOCK_SIZE + ty * THREAD_TILE + i;
                int bCol = col + j;
                s_B[ty * THREAD_TILE + i][tx * THREAD_TILE + j] = 
                    (bRow < d_N && bCol < d_N) ? __ldg(&B[bRow * d_N + bCol]) : 0.0f;
            }
        }

        __syncthreads();

        // Compute the multiplication for this tile with inner loop unrolling
        // Process the BLOCK_SIZE elements in steps of UNROLL_FACTOR
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += UNROLL_FACTOR) {
            #pragma unroll
            for (int u = 0; u < UNROLL_FACTOR; u++) {
                // Load two consecutive elements per thread from A and B
                float a0 = s_A[ty * THREAD_TILE + 0][k + u];
                float a1 = s_A[ty * THREAD_TILE + 1][k + u];
                float b0 = s_B[k + u][tx * THREAD_TILE + 0];
                float b1 = s_B[k + u][tx * THREAD_TILE + 1];

                regC[0][0] += a0 * b0;
                regC[0][1] += a0 * b1;
                regC[1][0] += a1 * b0;
                regC[1][1] += a1 * b1;
            }
        }

        __syncthreads();
    }

    // Write the computed 2x2 sub-tile from registers back into the global memory
    #pragma unroll
    for (int i = 0; i < THREAD_TILE; i++) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE; j++) {
            int outRow = row + i;
            int outCol = col + j;
            if (outRow < d_N && outCol < d_N) {
                C[outRow * d_N + outCol] = regC[i][j];
            }
        }
    }
}

// C++ interface (Pybind11 binding)
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_num_tiles, &num_tiles, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Launch configuration: each block covers a BLOCK_SIZE x BLOCK_SIZE output tile
    // Each thread computes a 2x2 sub-tile
    dim3 threads(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel_combined<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined Unrolled and Aligned Matrix Multiplication (CUDA)");
}
