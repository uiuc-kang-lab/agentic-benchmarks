#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Block size for rows and vectorized columns
#define BLOCK_SIZE 16
// Each thread computes 2 output elements in the horizontal (N) dimension
#define VECTOR_SIZE 2

// Kernel: Each thread computes a 2-element vector of the output matrix
// This vectorized approach distributes the workload evenly across threads and blocks,
// reducing underutilization in boundary cases and increasing arithmetic intensity.

template <typename scalar_t>
__global__ void matmul_vectorized_kernel(const scalar_t* __restrict__ A,
                                           const scalar_t* __restrict__ B,
                                           scalar_t* __restrict__ C,
                                           int M, int K, int N) {
    // Compute the global row index for C
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    // Compute the starting column index for C that this thread is responsible for
    int col_start = blockIdx.x * (BLOCK_SIZE * VECTOR_SIZE) + threadIdx.x * VECTOR_SIZE;

    // Registers for accumulating the result for two output elements per thread
    scalar_t sum0 = 0;
    scalar_t sum1 = 0;

    // Calculate number of tiles in the K dimension
    int numTiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate shared memory for tiles of A and B
    __shared__ scalar_t sA[BLOCK_SIZE][BLOCK_SIZE];
    // For B, each block loads BLOCK_SIZE rows and BLOCK_SIZE*VECTOR_SIZE columns
    __shared__ scalar_t sB[BLOCK_SIZE][BLOCK_SIZE * VECTOR_SIZE];

    // Loop over tiles along the K dimension
    for (int t = 0; t < numTiles; t++) {
        // Global column index in A for loading into shared memory
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        if (row < M && a_col < K)
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + a_col]);
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        // For B: each thread loads two elements per row for the tile
        int b_row = t * BLOCK_SIZE + threadIdx.y;
        int b_col0 = col_start;       // first column for this thread
        int b_col1 = col_start + 1;     // second column for this thread
        if (b_row < K) {
            if (b_col0 < N)
                sB[threadIdx.y][threadIdx.x * VECTOR_SIZE] = __ldg(&B[b_row * N + b_col0]);
            else
                sB[threadIdx.y][threadIdx.x * VECTOR_SIZE] = 0;

            if (b_col1 < N)
                sB[threadIdx.y][threadIdx.x * VECTOR_SIZE + 1] = __ldg(&B[b_row * N + b_col1]);
            else
                sB[threadIdx.y][threadIdx.x * VECTOR_SIZE + 1] = 0;
        } else {
            sB[threadIdx.y][threadIdx.x * VECTOR_SIZE] = 0;
            sB[threadIdx.y][threadIdx.x * VECTOR_SIZE + 1] = 0;
        }

        __syncthreads();

        // Compute the partial dot product for the current tile
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE; i++) {
            sum0 += sA[threadIdx.y][i] * sB[i][threadIdx.x * VECTOR_SIZE];
            sum1 += sA[threadIdx.y][i] * sB[i][threadIdx.x * VECTOR_SIZE + 1];
        }
        __syncthreads();
    }

    // Write the computed results back to C
    if (row < M && col_start < N)
        C[row * N + col_start] = sum0;
    if (row < M && (col_start + 1) < N)
        C[row * N + (col_start + 1)] = sum1;
}

// Host function exposed via Pybind11
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    // Define block and grid dimensions
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    // Each block covers BLOCK_SIZE rows and BLOCK_SIZE*VECTOR_SIZE columns
    dim3 blocks((N + BLOCK_SIZE * VECTOR_SIZE - 1) / (BLOCK_SIZE * VECTOR_SIZE),
                (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_vectorized_kernel", ([&] {
        matmul_vectorized_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    }));

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Vectorized matmul with workload distribution (CUDA)");
}
