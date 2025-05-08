#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <algorithm>

#define TILE_SIZE 32

// Kernel that computes C = A.T * B for a partition of rows of C.
// A: shape (K, M) stored in row-major order
// B: shape (K, N) stored in row-major order
// C: shape (M, N) stored in row-major order, where each element C[i,j] = sum_{k=0}^{K-1} A[k*M + i] * B[k*N + j]
// row_offset: the starting row index (i) for this partition
__global__ void matMulKernelPartition(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int K, int M, int N,
                                        int row_offset) {
    // local row index within this partition
    int local_row = blockIdx.x * TILE_SIZE + threadIdx.y;
    // global row index in C (and column index in A)
    int global_row = row_offset + local_row;
    int col = blockIdx.y * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Number of tiles needed to cover the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Each thread loads one element for tileA
        int aIndex = t * TILE_SIZE + threadIdx.x;  // k index for A
        if (global_row < M && aIndex < K) {
            // Note: A is stored as (K, M), so element for A.T at (global_row, aIndex) comes from A[aIndex * M + global_row]
            tileA[threadIdx.y][threadIdx.x] = A[aIndex * M + global_row];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Each thread loads one element for tileB
        int bIndex = t * TILE_SIZE + threadIdx.y;  // k index for B
        if (bIndex < K && col < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bIndex * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for the tile
        #pragma unroll
        for (int k_inner = 0; k_inner < TILE_SIZE; k_inner++) {
            sum += tileA[threadIdx.y][k_inner] * tileB[k_inner][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (global_row < M && col < N) {
        C[global_row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11.
// This version partitions the output matrix along the M dimension and launches concurrent kernel streams.
// Overlapping kernel execution and memory operations among streams can hide some latency and improve throughput.
// Input A: Tensor with shape (K, M), B: Tensor with shape (K, N).
// Output: Tensor C with shape (M, N) computed as C = A.T * B.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors of type float32
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor using torch::empty to avoid the cost of zero initialization
    auto C = torch::empty({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Use multiple CUDA streams to partition the work and overlap memory operations with computation.
    const int num_streams = 2;  // Can be tuned further
    std::vector<cudaStream_t> streams(num_streams);
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Partition the M dimension (rows of C) among the available streams.
    int rows_per_partition = (M + num_streams - 1) / num_streams;

    // Launch the kernel for each partition on its own stream
    for (int s = 0; s < num_streams; s++) {
        int row_offset = s * rows_per_partition;
        int rows_in_partition = std::min(rows_per_partition, M - row_offset);
        if (rows_in_partition <= 0) continue;

        // Compute grid dimensions based on the number of rows in this partition and full N
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((rows_in_partition + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

        matMulKernelPartition<<<gridDim, blockDim, 0, streams[s]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            K, M, N,
            row_offset
        );
    }

    // Synchronize and destroy streams
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using partitioned kernel launches with CUDA streams to overlap computation and memory transfers");
}
