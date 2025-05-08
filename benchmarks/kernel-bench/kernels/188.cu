#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <iostream>

#define TILE_SIZE 16

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tiled matrix multiplication kernel operating on a submatrix (chunk) of A and C
__global__ void matmul_kernel(const float* A, const float* B, float* C, int M_chunk, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Compute local row and column indices within the chunk
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles in the K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // Load tile from A (for current chunk) into shared memory
        if (row < M_chunk && aCol < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        else
            tileA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B into shared memory
        if (bRow < K && col < N)
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the computed value to C if within bounds
    if (row < M_chunk && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function partitions the matrix multiplication task along the row dimension.
// Each chunk of rows from A and C is processed on its own CUDA stream, overlapping kernel execution
// with memory operations to better utilize the GPU's resources (and avoid unnecessary initialization overhead
// by using torch::empty for the output).

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor without initializing values (since the kernel writes to all elements).
    torch::Tensor C = torch::empty({M, N}, A.options());

    // Partition the work along the row dimension of A and C.
    // Here we choose a chunk size (number of rows per kernel launch) that can be tuned.
    int chunk_size = 128;
    int num_chunks = (M + chunk_size - 1) / chunk_size;

    std::vector<cudaStream_t> streams(num_chunks);
    for (int i = 0; i < num_chunks; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // B remains the same across all chunks
    const float* B_ptr = B.data_ptr<float>();

    // Launch one kernel per chunk asynchronously
    for (int i = 0; i < num_chunks; i++) {
        int current_chunk_size = ((i + 1) * chunk_size > M) ? (M - i * chunk_size) : chunk_size;
        // Pointer offset for the current chunk for A and C
        const float* A_ptr = A.data_ptr<float>() + i * chunk_size * K;
        float* C_ptr = C.data_ptr<float>() + i * chunk_size * N;

        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (current_chunk_size + TILE_SIZE - 1) / TILE_SIZE);

        // Launch the kernel on the current stream
        matmul_kernel<<<grid, block, 0, streams[i]>>>(A_ptr, B_ptr, C_ptr, current_chunk_size, N, K);
    }

    // Synchronize and destroy all CUDA streams
    for (int i = 0; i < num_chunks; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with overlapping computation and memory transfers (CUDA)");
}
