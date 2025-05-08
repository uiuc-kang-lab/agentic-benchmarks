#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Tile and block dimensions
#define TILE_M 16    // Tile width for output (also for A's column index)
#define TILE_N 16    // Tile height for output (also for B's column index)
#define BLOCK_K 32   // Chunk size along the K dimension
#define NUM_STAGES 2 // Double buffering stages

// This kernel uses asynchronous copy instructions (cp.async) to overlap
// the loading of data from global memory into shared memory with computation.
// It pipelines the transfers using double buffering in shared memory, with the
// aim of hiding global memory latency. The kernel computes C = A.T * B where
// A is of shape (K, M) and B is of shape (K, N) and the result C has shape (M, N).

__global__ void pipelinedTiledSharedKernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int K, int M, int N) {
    // Calculate global indices for output matrix C
    int row = blockIdx.x * TILE_M + threadIdx.x;
    int col = blockIdx.y * TILE_N + threadIdx.y;
    float sum = 0.0f;

    // Allocate double-buffered shared memory for tiles of A and B
    __shared__ float As[NUM_STAGES][BLOCK_K][TILE_M];
    __shared__ float Bs[NUM_STAGES][BLOCK_K][TILE_N];

    int numTiles = (K + BLOCK_K - 1) / BLOCK_K;
    int current_buf = 0;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int totalThreads = blockDim.x * blockDim.y;

    // Initiate asynchronous copy for the first tile (tile 0), if available
    if(numTiles > 0) {
        // Load tile 0 for matrix A into As[current_buf]
        for (int i = tid; i < BLOCK_K * TILE_M; i += totalThreads) {
            int t = i / TILE_M;
            int i_local = i % TILE_M;
            int global_k = t; // tile 0
            int global_i = blockIdx.x * TILE_M + i_local;
            if(global_k < K && global_i < M) {
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], %2;\n"
                    :
                    : "r"(&As[current_buf][t][i_local]),
                      "l"(&A[global_k * M + global_i]),
                      "n"(sizeof(float))
                );
            } else {
                As[current_buf][t][i_local] = 0.0f;
            }
        }
        // Load tile 0 for matrix B into Bs[current_buf]
        for (int i = tid; i < BLOCK_K * TILE_N; i += totalThreads) {
            int t = i / TILE_N;
            int j_local = i % TILE_N;
            int global_k = t; // tile 0
            int global_j = blockIdx.y * TILE_N + j_local;
            if(global_k < K && global_j < N) {
                asm volatile(
                    "cp.async.cg.shared.global [%0], [%1], %2;\n"
                    :
                    : "r"(&Bs[current_buf][t][j_local]),
                      "l"(&B[global_k * N + global_j]),
                      "n"(sizeof(float))
                );
            } else {
                Bs[current_buf][t][j_local] = 0.0f;
            }
        }
        // Commit the async copy group and wait for completion
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
    }
    __syncthreads();

    // Pipeline over tiles
    for (int tile = 0; tile < numTiles; tile++) {
        int next_tile = tile + 1;
        int next_buf = 1 - current_buf;
        if (next_tile < numTiles) {
            // Asynchronously load the next tile for matrix A into next_buf
            for (int i = tid; i < BLOCK_K * TILE_M; i += totalThreads) {
                int t = i / TILE_M;
                int i_local = i % TILE_M;
                int global_k = next_tile * BLOCK_K + t;
                int global_i = blockIdx.x * TILE_M + i_local;
                if(global_k < K && global_i < M) {
                    asm volatile(
                        "cp.async.cg.shared.global [%0], [%1], %2;\n"
                        :
                        : "r"(&As[next_buf][t][i_local]),
                          "l"(&A[global_k * M + global_i]),
                          "n"(sizeof(float))
                    );
                } else {
                    As[next_buf][t][i_local] = 0.0f;
                }
            }
            // Asynchronously load the next tile for matrix B into next_buf
            for (int i = tid; i < BLOCK_K * TILE_N; i += totalThreads) {
                int t = i / TILE_N;
                int j_local = i % TILE_N;
                int global_k = next_tile * BLOCK_K + t;
                int global_j = blockIdx.y * TILE_N + j_local;
                if(global_k < K && global_j < N) {
                    asm volatile(
                        "cp.async.cg.shared.global [%0], [%1], %2;\n"
                        :
                        : "r"(&Bs[next_buf][t][j_local]),
                          "l"(&B[global_k * N + global_j]),
                          "n"(sizeof(float))
                    );
                } else {
                    Bs[next_buf][t][j_local] = 0.0f;
                }
            }
            asm volatile("cp.async.commit_group;\n");
        }
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        // Compute the partial dot product for the current tile using data in current_buf
        #pragma unroll
        for (int t = 0; t < BLOCK_K; t += 4) {
            sum += As[current_buf][t][threadIdx.x] * Bs[current_buf][t][threadIdx.y];
            if(t + 1 < BLOCK_K)
                sum += As[current_buf][t+1][threadIdx.x] * Bs[current_buf][t+1][threadIdx.y];
            if(t + 2 < BLOCK_K)
                sum += As[current_buf][t+2][threadIdx.x] * Bs[current_buf][t+2][threadIdx.y];
            if(t + 3 < BLOCK_K)
                sum += As[current_buf][t+3][threadIdx.x] * Bs[current_buf][t+3][threadIdx.y];
        }
        __syncthreads();
        current_buf = (next_tile < numTiles) ? next_buf : current_buf;
    }

    // Write the computed result to global memory
    if (row < M && col < N)
        C[row * N + col] = sum;
}

// Forward function exposed via PyBind11
// It retrieves the current CUDA stream to allow overlapping kernel execution with memory operations.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define block and grid dimensions
    dim3 block(TILE_M, TILE_N);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    // Retrieve the current CUDA stream to enable overlapping of kernel execution and memory operations
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pipelinedTiledSharedKernel<<<grid, block, 0, stream>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using pipelined tiled shared memory with asynchronous copies (CUDA)");
}
