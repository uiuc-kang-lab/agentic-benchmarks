/*
Hybrid CUDA kernel for Matrix Multiplication
Combines shared-memory tiling with an optional 3D grid partition along the K-dimension.
When the K dimension is large, the kernel is launched with gridDim.z > 1, so that different blocks (with different blockIdx.z) work on different k-tiles and use atomicAdd to accumulate partial results.
When gridDim.z == 1, the kernel uses a loop over all k-tiles and writes the result directly, avoiding atomic overhead.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define block size for M and N dimensions and tile size for K dimension
#define BLOCK_SIZE 16
#define TILE_K 16

// Device function to access an element with optional transpose
__device__ __forceinline__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

// Hybrid kernel: Uses a 3D grid for partitioning the K dimension and shared memory tiling
__global__ void matmul_hybrid_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K,
                                       int lda, int ldb, int ldc,
                                       bool transA, bool transB) {
    // Compute global row and column indices
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Total number of K-tiles
    int numTiles = (K + TILE_K - 1) / TILE_K;

    // Loop over k tiles in a strided manner along gridDim.z
    for (int t = blockIdx.z; t < numTiles; t += gridDim.z) {
        int k_start = t * TILE_K;

        // Declare shared memory tiles for A and B
        __shared__ float As[BLOCK_SIZE][TILE_K];
        __shared__ float Bs[TILE_K][BLOCK_SIZE];

        // Load a tile of A into shared memory
        int a_col = k_start + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = get_element(A, row, a_col, lda, transA);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of B into shared memory
        int b_row = k_start + threadIdx.y;
        if (col < N && b_row < K) {
            Bs[threadIdx.y][threadIdx.x] = get_element(B, b_row, col, ldb, transB);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles together
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N) {
        // If gridDim.z > 1, multiple blocks contribute to the same C element
        if (gridDim.z > 1) {
            atomicAdd(&C[row * ldc + col], sum);
        } else {
            C[row * ldc + col] = sum;
        }
    }
}

// Host function interfacing with PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check for CUDA tensors and correct dimensionality
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Get dimensions and decide on transpose flags based on memory layout
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int64_t M, N, K;
    int lda, ldb, ldc;

    if (A_rows >= A_cols && B_rows == A_cols) {
        // A is MxK, B is KxN
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A is stored transposed
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B is stored transposed
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Both A and B are stored transposed
        transA = true;
        transB = true;
        M = A_cols;
        K = A_rows;
        N = B_rows;
        lda = A.stride(1);
        ldb = B.stride(1);
    } else {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }

    ldc = N;

    // Determine the grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    int grid_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_z = (K + TILE_K - 1) / TILE_K; // Partition K-dimension
    grid_z = (grid_z < 1) ? 1 : grid_z;
    dim3 gridDim(grid_x, grid_y, grid_z);

    // Allocate output tensor
    // If using atomic accumulation (gridDim.z > 1), initialize C with zeros
    auto C = (gridDim.z > 1) ? torch::zeros({M, N}, A.options())
                             : torch::empty({M, N}, A.options());

    // Launch the kernel
    matmul_hybrid_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Hybrid matrix multiplication with shared-memory tiling and optional K-dimension partitioning (CUDA)");
}
