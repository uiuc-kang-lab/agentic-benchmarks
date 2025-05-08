/*
Combined CUDA Kernel that integrates shared memory bank conflict avoidance,
loop unrolling, and multi-streamed execution for improved performance.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

#define BLOCK_SIZE 32
#define NUM_STREAMS 4

// Combined kernel that processes a chunk of rows using shared memory tiling with bank conflict padding
__global__ void matmul_combined_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         const int N,
                                         const int chunk_start,
                                         const int chunk_size) {
    // Use padding in shared memory to avoid bank conflicts
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE + 1];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Calculate global row index with chunk offset and column index
    int row = blockIdx.y * BLOCK_SIZE + ty + chunk_start;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Loop over all tiles
    for (int t = 0; t < numTiles; t++) {
        int tileOffset = t * BLOCK_SIZE;

        // Load tile from A with coalesced access and boundary check
        if (row < chunk_start + chunk_size && (tileOffset + tx) < N)
            s_A[ty][tx] = A[row * N + tileOffset + tx];
        else
            s_A[ty][tx] = 0.0f;

        // Load tile from B with coalesced access and boundary check
        if (col < N && (tileOffset + ty) < N)
            s_B[ty][tx] = B[(tileOffset + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product with unrolled computation and register caching (factor of 4)
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += 4) {
            float a0 = s_A[ty][k];
            float a1 = s_A[ty][k + 1];
            float a2 = s_A[ty][k + 2];
            float a3 = s_A[ty][k + 3];
            float b0 = s_B[k][tx];
            float b1 = s_B[k + 1][tx];
            float b2 = s_B[k + 2][tx];
            float b3 = s_B[k + 3][tx];
            sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        }
        
        __syncthreads();
    }

    // Write the result if in bounds
    if (row < chunk_start + chunk_size && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function that orchestrates the multi-stream execution
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Initialize CUDA streams for parallel chunk processing
    std::vector<cudaStream_t> streams(NUM_STREAMS);
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }

    // Divide the rows into chunks, one for each stream
    int chunk_size = (N + NUM_STREAMS - 1) / NUM_STREAMS;
    
    // Launch the kernel for each stream on its assigned chunk
    for (int i = 0; i < NUM_STREAMS; i++) {
        int chunk_start = i * chunk_size;
        int current_chunk_size = std::min(chunk_size, N - chunk_start);
        if (current_chunk_size <= 0) break;

        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (current_chunk_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

        matmul_combined_kernel<<<blocks, threads, 0, streams[i]>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            chunk_start,
            current_chunk_size
        );
    }

    // Synchronize and destroy all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined Streamed and Coalesced Matrix Multiplication (CUDA)");
}
