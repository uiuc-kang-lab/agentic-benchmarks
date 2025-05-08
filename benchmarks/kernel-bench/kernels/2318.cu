#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Blocking parameters
#define BLOCK_M 64
#define BLOCK_N 64
#define TILE_K 16

// Register tile sizes per thread
#define REG_TILE_M 4
#define REG_TILE_N 4

// Kernel: Each thread block computes a 64x64 tile of the output matrix C.
// Each thread computes a 4x4 sub-tile, ensuring an even distribution of work across threads.
// C = A * B^T, where A is (M x K) and B is (N x K). Note that B's row corresponds to the output column.

__global__ void matmul_tiled_kernel(const float* __restrict__ A, 
                                    const float* __restrict__ B, 
                                    float* __restrict__ C, 
                                    int M, int N, int K) {
    // Shared memory for A and B tiles
    __shared__ float As[BLOCK_M][TILE_K];
    __shared__ float Bs[BLOCK_N][TILE_K];

    // Block and thread indices
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int thread_row = threadIdx.y; // range: 0 to blockDim.y-1
    int thread_col = threadIdx.x; // range: 0 to blockDim.x-1

    // Each thread will compute a REG_TILE_M x REG_TILE_N sub-tile
    float acc[REG_TILE_M][REG_TILE_N];
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; j++) {
            acc[i][j] = 0.0f;
        }
    }

    // Starting indices for the sub-tile computed by this thread
    int row_start = block_row * BLOCK_M + thread_row * REG_TILE_M;
    int col_start = block_col * BLOCK_N + thread_col * REG_TILE_N;

    // Loop over tiles of K
    int numTiles = (K + TILE_K - 1) / TILE_K;
    for (int t = 0; t < numTiles; t++) {
        int k_start = t * TILE_K;

        // Load A tile into shared memory
        // Total elements in the A tile: BLOCK_M * TILE_K
        int numElementsA = BLOCK_M * TILE_K;
        int threadId = thread_row * blockDim.x + thread_col; // linear thread index in block
        for (int idx = threadId; idx < numElementsA; idx += blockDim.x * blockDim.y) {
            int i = idx / TILE_K; // row index within the A tile
            int j = idx % TILE_K; // col index within the tile
            int a_row = block_row * BLOCK_M + i;
            int a_col = k_start + j;
            if (a_row < M && a_col < K)
                As[i][j] = A[a_row * K + a_col];
            else
                As[i][j] = 0.0f;
        }

        // Load B tile into shared memory
        // For B, note that C[m, n] = sum_k A[m,k] * B[n,k] since C = A * B^T.
        // So we load a tile of B corresponding to rows (block_col * BLOCK_N, block_col * BLOCK_N + BLOCK_N)
        int numElementsB = BLOCK_N * TILE_K;
        for (int idx = threadId; idx < numElementsB; idx += blockDim.x * blockDim.y) {
            int i = idx / TILE_K; // row index within the B tile
            int j = idx % TILE_K; // col index within the tile
            int b_row = block_col * BLOCK_N + i; // B's row corresponds to output column index
            int b_col = k_start + j;
            if (b_row < N && b_col < K)
                Bs[i][j] = B[b_row * K + b_col];
            else
                Bs[i][j] = 0.0f;
        }

        __syncthreads();

        // Compute partial results for the sub-tile using the loaded tile
        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_K; k_inner++) {
            // Load a vector of REG_TILE_M elements from A tile
            float a_vals[REG_TILE_M];
            #pragma unroll
            for (int i = 0; i < REG_TILE_M; i++) {
                int a_index = thread_row * REG_TILE_M + i; // index in As
                a_vals[i] = As[a_index][k_inner];
            }
            
            // Load a vector of REG_TILE_N elements from B tile
            float b_vals[REG_TILE_N];
            #pragma unroll
            for (int j = 0; j < REG_TILE_N; j++) {
                int b_index = thread_col * REG_TILE_N + j; // index in Bs
                b_vals[j] = Bs[b_index][k_inner];
            }
            
            // Multiply and accumulate
            #pragma unroll
            for (int i = 0; i < REG_TILE_M; i++) {
                #pragma unroll
                for (int j = 0; j < REG_TILE_N; j++) {
                    acc[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }

        __syncthreads();
    }

    // Write the computed sub-tile to global memory
    #pragma unroll
    for (int i = 0; i < REG_TILE_M; i++) {
        #pragma unroll
        for (int j = 0; j < REG_TILE_N; j++) {
            int global_row = block_row * BLOCK_M + thread_row * REG_TILE_M + i;
            int global_col = block_col * BLOCK_N + thread_col * REG_TILE_N + j;
            if (global_row < M && global_col < N) {
                C[global_row * N + global_col] = acc[i][j];
            }
        }
    }
}


// C = A * B^T, where A is (M x K) and B is (N x K), resulting in C (M x N)
// Even workload distribution is achieved by having each thread compute a REG_TILE_M x REG_TILE_N block
// and by cooperatively loading tiles from global memory into shared memory.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Launch configuration
    // Block dimensions: 16 x 16 threads per block
    // Each block computes a 64x64 tile of C
    dim3 block(16, 16);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);

    matmul_tiled_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B using tiled shared memory and register blocking (CUDA)");
}
