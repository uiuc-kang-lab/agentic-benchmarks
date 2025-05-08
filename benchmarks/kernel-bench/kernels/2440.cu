#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Define tile sizes for output and K-dimension partition
#define TILE_M 16  // Number of rows in a tile
#define TILE_N 16  // Number of columns in a tile
#define TILE_K 16  // Chunk size for splitting the K dimension

// Kernel using split-K parallelism and atomicAdd to accumulate partial results
// Each block computes a partial sum for one tile of the output matrix C over a segment of K
// A is (K x M): accessed as A[k * M + m]
// B is (N x K): accessed as B[n * K + k]
// C is (M x N): accessed as C[m * N + n]
// It is assumed that C is initialized to zero before accumulation

template <typename scalar_t>
__global__ void matmul_transpose_atomic_split_kernel(
    const scalar_t* __restrict__ A, // A: (K x M)
    const scalar_t* __restrict__ B, // B: (N x K)
    scalar_t* __restrict__ C,       // C: (M x N), prezeroed
    int M, int N, int K) {

    // 2D block for output tile and third dimension for split-K
    int block_m = blockIdx.y; // Tile row index for C
    int block_n = blockIdx.x; // Tile column index for C
    int split_k = blockIdx.z; // Partition index along K dimension

    // Global output element indices
    int row = block_m * TILE_M + threadIdx.x;  // row index in C (m)
    int col = block_n * TILE_N + threadIdx.y;    // column index in C (n)

    // Determine the K range this block will process
    int k_start = split_k * TILE_K;
    int k_end = (k_start + TILE_K < K) ? (k_start + TILE_K) : K;

    // Shared memory tiles for A and B
    // For A: dimensions TILE_K x TILE_M; for B: dimensions TILE_N x TILE_K
    __shared__ scalar_t A_tile[TILE_K][TILE_M];
    __shared__ scalar_t B_tile[TILE_N][TILE_K];

    // Cooperative loading of A_tile and B_tile
    int tId = threadIdx.y * blockDim.x + threadIdx.x;  // Flattened thread index within the block
    int totalThreads = blockDim.x * blockDim.y;           // Should be TILE_M * TILE_N (256 threads)

    // Load tile from A
    // A is stored as (K x M) with A[k * M + m]
    int totalAElements = TILE_K * TILE_M;
    for (int index = tId; index < totalAElements; index += totalThreads) {
        int i = index / TILE_M; // i in [0, TILE_K)
        int j = index % TILE_M; // j in [0, TILE_M)
        int global_m = block_m * TILE_M + j;
        int global_k = k_start + i;
        if (global_m < M && global_k < K) {
            A_tile[i][j] = __ldg(&A[global_k * M + global_m]);
        } else {
            A_tile[i][j] = static_cast<scalar_t>(0);
        }
    }

    // Load tile from B
    // B is stored as (N x K) with B[n * K + k]
    int totalBElements = TILE_N * TILE_K;
    for (int index = tId; index < totalBElements; index += totalThreads) {
        int i = index / TILE_K; // i in [0, TILE_N)
        int j = index % TILE_K; // j in [0, TILE_K)
        int global_n = block_n * TILE_N + i;
        int global_k = k_start + j;
        if (global_n < N && global_k < K) {
            B_tile[i][j] = __ldg(&B[global_n * K + global_k]);
        } else {
            B_tile[i][j] = static_cast<scalar_t>(0);
        }
    }
    __syncthreads();

    // Each thread computes the partial dot product for its corresponding element of C
    scalar_t partial = 0;
    int effectiveK = k_end - k_start;  // May be less than TILE_K at the boundary
    for (int k = 0; k < effectiveK; k++) {
        // For the transposed multiplication, use A_tile[k][threadIdx.x] and B_tile[threadIdx.y][k]
        partial += A_tile[k][threadIdx.x] * B_tile[threadIdx.y][k];
    }
    __syncthreads();

    // Accumulate the partial result into global memory using atomicAdd to avoid race conditions
    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], partial);
    }
}

// PyTorch binding

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // Dimensions:
    // A: (K x M), B: (N x K) => C: (M x N)
    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(0);

    // Initialize output tensor C to zero as we accumulate partial sums
    auto C = torch::zeros({M, N}, A.options());

    // Grid dimensions: we partition C in 2D and split the K dimension in the third dimension
    dim3 threads(TILE_M, TILE_N);  // 16x16 threads per block
    dim3 blocks((N + TILE_N - 1) / TILE_N,
                (M + TILE_M - 1) / TILE_M,
                (K + TILE_K - 1) / TILE_K);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_atomic_split_kernel", ([&] {
        matmul_transpose_atomic_split_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using atomic split-K optimization (CUDA)");
}
