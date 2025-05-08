#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size and maximum constant memory size for matrix B.
// This kernel assumes that the entire B matrix (of size N x N) fits in constant memory.
// For 64 KB of constant memory, maximum N*N is 16384 floats (i.e. N <= 128).
#define TILE_SIZE 32
#define MAX_MATRIX_SIZE 16384  // Maximum number of floats in B

// Declare constant memory for matrix B. The kernel will use this for read-only access.
__constant__ float const_B[MAX_MATRIX_SIZE];

// Optimized kernel for lower triangular matrix multiplication using constant memory for matrix B
// and shared memory tiling for matrix A. It computes C = tril(A * B), where for each (row, col) with row >= col:
//     C[row, col] = sum_{k=col}^{row} A[row, k] * B[k, col]
// Elements in the upper triangular portion (row < col) are set to 0.
__global__ void constant_mem_triangular_mm_kernel(const float* __restrict__ A,
                                                    float* __restrict__ C,
                                                    int N) {
    // Compute global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Return if out-of-bound
    if (row >= N || col >= N) return;

    // Early block-level pruning: if the entire block is in the upper triangular region,
    // then all elements in this block are zero.
    int block_row_max = blockIdx.y * TILE_SIZE + TILE_SIZE - 1;
    int block_col_min = blockIdx.x * TILE_SIZE;
    if (block_row_max < block_col_min) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;

    // Shared memory tile for matrix A
    __shared__ float sA[TILE_SIZE][TILE_SIZE];

    // Process the multiplication in tiles along the k-dimension
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int m = 0; m < numTiles; m++) {
        int k = m * TILE_SIZE + threadIdx.x;  // global column index for A
        // Load A[row, k] into shared memory if within bounds and only if k <= row (since A is lower triangular)
        if (k < N && row >= k)
            sA[threadIdx.y][threadIdx.x] = A[row * N + k];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Determine the valid range of k indices in this tile that contribute to the sum
        int tile_start = m * TILE_SIZE;
        int tile_end = tile_start + TILE_SIZE;
        if (tile_end > N) tile_end = N;

        // For lower triangular multiplication, the summation index k must satisfy: col <= k <= row.
        int valid_start = (col > tile_start) ? col : tile_start;
        int valid_end = (row + 1 < tile_end) ? row + 1 : tile_end;

        int local_start = valid_start - tile_start;  // corresponding index in shared memory tile
        int local_end = valid_end - tile_start;

        // Unroll the inner loop over the shared tile
        #pragma unroll
        for (int i = local_start; i < local_end; i++) {
            // Load B[k, col] from constant memory; here, k = tile_start + i
            sum += sA[threadIdx.y][i] * const_B[(tile_start + i) * N + col];
        }

        __syncthreads();
    }

    // Ensure that elements in the upper triangular region are zero
    if (row < col) sum = 0.0f;
    C[row * N + col] = sum;
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    // Ensure the size of B fits in constant memory
    TORCH_CHECK(N * N <= MAX_MATRIX_SIZE, "Matrix too large for constant memory optimization");

    auto C = torch::empty_like(A);

    // Copy matrix B into constant memory. We use cudaMemcpyToSymbol with DeviceToDevice copy,
    // since B is already a CUDA tensor.
    cudaError_t err = cudaMemcpyToSymbol(const_B, B.data_ptr<float>(), N * N * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpyToSymbol failed: ", cudaGetErrorString(err));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    constant_mem_triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Constant Memory Optimized Lower Triangular Matrix Multiplication (CUDA)");
}
