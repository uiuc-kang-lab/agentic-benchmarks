#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

#define TILE_WIDTH 16
const int NUM_STREAMS = 4;

// Combined kernel using tiling, shared memory, __ldg read-only loads, and multi-stream processing

template <typename scalar_t>
__global__ void matmul_stream_ldg_kernel(const scalar_t* __restrict__ A,
                                           const scalar_t* __restrict__ B,
                                           scalar_t* __restrict__ C,
                                           int K, int N, int row_offset, int valid_rows) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    // Compute global row and col indices
    int row = row_offset + blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    scalar_t value = 0;
    int num_tiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < num_tiles; ++t) {
        int tiledA_col = t * TILE_WIDTH + threadIdx.x;
        int tiledB_row = t * TILE_WIDTH + threadIdx.y;
        
        // Load tile from A using __ldg if within bounds
        if (row < (row_offset + valid_rows) && tiledA_col < K)
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tiledA_col]);
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        // Load tile from B using __ldg if within bounds
        if (col < N && tiledB_row < K)
            sB[threadIdx.y][threadIdx.x] = __ldg(&B[tiledB_row * N + col]);
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }
        
        __syncthreads();
    }

    // Write output if within bounds
    if (row < (row_offset + valid_rows) && col < N)
        C[row * N + col] = value;
}

// Host function that splits the workload across multiple CUDA streams

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    // Allocate output tensor
    auto C = torch::empty({M, N}, A.options());

    // Create multiple streams to process chunks of rows concurrently
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreate(&streams[i]);

    int chunk_size = (M + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_stream_ldg_kernel", ([&] {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            int row_offset = s * chunk_size;
            int valid_rows = std::min(chunk_size, static_cast<int>(M - row_offset));
            if (valid_rows <= 0) break;

            dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                           (valid_rows + TILE_WIDTH - 1) / TILE_WIDTH);
            
            matmul_stream_ldg_kernel<scalar_t><<<numBlocks, threadsPerBlock, 0, streams[s]>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                K, N, row_offset, valid_rows
            );
        }
    }));

    // Synchronize and destroy streams
    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Multi-stream and __ldg optimized matrix multiplication");
}
