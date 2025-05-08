#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
const int NUM_STREAMS = 4;

// Device function to load a tile from matrix A into shared memory using __ldg()
template <typename scalar_t>
__device__ inline void load_tile(const scalar_t* __restrict__ matrix,
                                 scalar_t tile[TILE_WIDTH][TILE_WIDTH],
                                 int row, int col, int stride, int max_row, int max_col) {
    if (row < max_row && col < max_col)
        tile[threadIdx.y][threadIdx.x] = __ldg(&matrix[row * stride + col]);
    else
        tile[threadIdx.y][threadIdx.x] = 0;
}

// Device function to compute the dot product for a tile
template <typename scalar_t>
__device__ inline scalar_t compute_tile(const scalar_t A_tile[TILE_WIDTH][TILE_WIDTH],
                                        const scalar_t B_tile[TILE_WIDTH][TILE_WIDTH]) {
    scalar_t sum = 0;
    #pragma unroll
    for (int k = 0; k < TILE_WIDTH; ++k) {
        sum += A_tile[threadIdx.y][k] * B_tile[k][threadIdx.x];
    }
    return sum;
}

// Combined CUDA kernel for matrix multiplication with streams and tiling
template <typename scalar_t>
__global__ void matmul_optimized_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                        scalar_t* __restrict__ C, int M, int K, int N, int row_offset) {
    __shared__ scalar_t A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t B_tile[TILE_WIDTH][TILE_WIDTH];

    int row = row_offset + blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t value = 0;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; ++t) {
        load_tile<scalar_t>(A, A_tile, row, t * TILE_WIDTH + threadIdx.x, K, row_offset + M, K);
        load_tile<scalar_t>(B, B_tile, t * TILE_WIDTH + threadIdx.y, col, N, K, N);
        __syncthreads();
        value += compute_tile<scalar_t>(A_tile, B_tile);
        __syncthreads();
    }

    if (row < (row_offset + M) && col < N)
        C[row * N + col] = value;
}

// Host function exposed to Python via Pybind11
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreate(&streams[i]);

    const int chunk_size = (M + NUM_STREAMS - 1) / NUM_STREAMS;
    dim3 threads(TILE_WIDTH, TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_optimized_kernel", [&] {
        for (int s = 0; s < NUM_STREAMS; ++s) {
            const int row_start = s * chunk_size;
            const int valid_rows = std::min(chunk_size, static_cast<int>(M - row_start));
            if (valid_rows <= 0) break;

            dim3 blocks((N + TILE_WIDTH-1)/TILE_WIDTH, (valid_rows + TILE_WIDTH-1)/TILE_WIDTH);
            matmul_optimized_kernel<scalar_t><<<blocks, threads, 0, streams[s]>>>(
                A.data_ptr<scalar_t>(),
                B.data_ptr<scalar_t>(),
                C.data_ptr<scalar_t>(),
                valid_rows,
                K,
                N,
                row_start
            );
        }
    });

    for (int s = 0; s < NUM_STREAMS; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Optimized streamed and tiled matrix multiplication (CUDA)");
}
