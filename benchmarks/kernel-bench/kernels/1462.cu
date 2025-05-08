#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define BLOCK_SIZE 32
#define THREAD_TILE 2
#define NUM_STREAMS 4
#define CHUNK_SIZE 1024
#define MAX_MATRIX_DIM 8192

// Constant memory for matrix dimension
__constant__ int d_N;

// Hybrid kernel: combines efficient __ldg()-based aligned loads and 2x2 tiling with stream-based chunk processing
__global__ void matmul_kernel_stream_aligned(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int chunk_start,
                                              int chunk_size) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // Compute global row (offset by chunk_start) and column indices for the 2x2 sub-tile
    int row = chunk_start + blockIdx.y * BLOCK_SIZE + ty * THREAD_TILE;
    int col = blockIdx.x * BLOCK_SIZE + tx * THREAD_TILE;

    // Registers for 2x2 sub-tile accumulation
    float regC00 = 0.0f, regC01 = 0.0f, regC10 = 0.0f, regC11 = 0.0f;

    int num_tiles = (d_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < num_tiles; t++) {
        // Load a 2x2 sub-tile from A into shared memory using __ldg for read-only caching
        int aRow0 = row;
        int aRow1 = row + 1;
        int aCol0 = t * BLOCK_SIZE + tx * THREAD_TILE;
        int aCol1 = aCol0 + 1;

        s_A[ty * THREAD_TILE + 0][tx * THREAD_TILE + 0] = (aRow0 < d_N && aCol0 < d_N) ? __ldg(&A[aRow0 * d_N + aCol0]) : 0.0f;
        s_A[ty * THREAD_TILE + 0][tx * THREAD_TILE + 1] = (aRow0 < d_N && aCol1 < d_N) ? __ldg(&A[aRow0 * d_N + aCol1]) : 0.0f;
        s_A[ty * THREAD_TILE + 1][tx * THREAD_TILE + 0] = (aRow1 < d_N && aCol0 < d_N) ? __ldg(&A[aRow1 * d_N + aCol0]) : 0.0f;
        s_A[ty * THREAD_TILE + 1][tx * THREAD_TILE + 1] = (aRow1 < d_N && aCol1 < d_N) ? __ldg(&A[aRow1 * d_N + aCol1]) : 0.0f;

        // Load a 2x2 sub-tile from B into shared memory
        int bRow0 = t * BLOCK_SIZE + ty * THREAD_TILE;
        int bRow1 = bRow0 + 1;
        int bCol0 = col;
        int bCol1 = col + 1;

        s_B[ty * THREAD_TILE + 0][tx * THREAD_TILE + 0] = (bRow0 < d_N && bCol0 < d_N) ? __ldg(&B[bRow0 * d_N + bCol0]) : 0.0f;
        s_B[ty * THREAD_TILE + 0][tx * THREAD_TILE + 1] = (bRow0 < d_N && bCol1 < d_N) ? __ldg(&B[bRow0 * d_N + bCol1]) : 0.0f;
        s_B[ty * THREAD_TILE + 1][tx * THREAD_TILE + 0] = (bRow1 < d_N && bCol0 < d_N) ? __ldg(&B[bRow1 * d_N + bCol0]) : 0.0f;
        s_B[ty * THREAD_TILE + 1][tx * THREAD_TILE + 1] = (bRow1 < d_N && bCol1 < d_N) ? __ldg(&B[bRow1 * d_N + bCol1]) : 0.0f;

        __syncthreads();

        // Multiply the loaded tile and accumulate into registers
        #pragma unroll
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

    // Write the 2x2 result back to global memory, ensuring within both the matrix and chunk boundaries
    if (row < d_N && row < chunk_start + chunk_size && col < d_N)
        C[row * d_N + col] = regC00;
    if (row < d_N && row < chunk_start + chunk_size && (col + 1) < d_N)
        C[row * d_N + col + 1] = regC01;
    if ((row + 1) < d_N && (row + 1) < chunk_start + chunk_size && col < d_N)
        C[(row + 1) * d_N + col] = regC10;
    if ((row + 1) < d_N && (row + 1) < chunk_start + chunk_size && (col + 1) < d_N)
        C[(row + 1) * d_N + col + 1] = regC11;
}


// Host function: launches the kernel using multiple CUDA streams for overlapping computation
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    dim3 threads(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);

    // Process the matrix in chunks along the row dimension using streams to overlap kernel execution
    for (int chunk_start = 0; chunk_start < N; chunk_start += CHUNK_SIZE) {
        int chunk_size = std::min(CHUNK_SIZE, N - chunk_start);
        int num_blocks_y = (chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        int num_blocks_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dim3 blocks(num_blocks_x, num_blocks_y);

        int stream_idx = (chunk_start / CHUNK_SIZE) % NUM_STREAMS;

        matmul_kernel_stream_aligned<<<blocks, threads, 0, streams[stream_idx]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            chunk_start,
            chunk_size
        );
    }

    // Synchronize and clean up streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stream-based and Aligned Matrix Multiplication (CUDA)");
}
