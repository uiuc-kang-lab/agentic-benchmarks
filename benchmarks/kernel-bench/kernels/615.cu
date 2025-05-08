#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
const int MAX_NUM_STREAMS = 8;  // Allowing flexibility in stream count based on device.

// Function to load a tile from matrix A into shared memory
template <typename scalar_t>
__device__ inline void load_A_tile(const scalar_t* __restrict__ A,
                                    scalar_t A_tile[TILE_WIDTH][TILE_WIDTH],
                                    int row, int tile, int M, int K) {
    int col = tile * TILE_WIDTH + threadIdx.x;
    if (row < M && col < K)
        A_tile[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + col]);
    else
        A_tile[threadIdx.y][threadIdx.x] = 0;
}

// Function to load a tile from matrix B into shared memory
template <typename scalar_t>
__device__ inline void load_B_tile(const scalar_t* __restrict__ B,
                                    scalar_t B_tile[TILE_WIDTH][TILE_WIDTH],
                                    int col, int tile, int K, int N) {
    int row = tile * TILE_WIDTH + threadIdx.y;
    if (row < K && col < N)
        B_tile[threadIdx.y][threadIdx.x] = __ldg(&B[row * N + col]);
    else
        B_tile[threadIdx.y][threadIdx.x] = 0;
}

// CUDA kernel using streams for matrix multiplication
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
        load_A_tile<scalar_t>(A, A_tile, row, t, M, K);
        load_B_tile<scalar_t>(B, B_tile, col, t, K, N);
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i)
            value += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];

        __syncthreads();
    }

    if (row < (row_offset + M) && col < N)
        C[row * N + col] = value;
}

// Host function exposed to Python via Pybind11
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B, int num_streams = 4) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(num_streams <= MAX_NUM_STREAMS && num_streams > 0, "Invalid number of streams");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Dimension mismatch");

    auto C = torch::empty({M, N}, A.options());

    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i)
        cudaStreamCreate(&streams[i]);

    const int chunk_size = (M + num_streams - 1) / num_streams;
    dim3 threads(TILE_WIDTH, TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_optimized_kernel", [&] {
        for (int s = 0; s < num_streams; ++s) {
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

    for (int s = 0; s < num_streams; ++s) {
        cudaStreamSynchronize(streams[s]);
        cudaStreamDestroy(streams[s]);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Optimized streaming matmul forward", py::arg("A"), py::arg("B"), py::arg("num_streams") = 4);
}
