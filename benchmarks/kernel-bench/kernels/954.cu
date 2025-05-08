#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define BLOCK_SIZE 16
#define TILE_K 16

// Device function to access elements with optional transpose
__device__ __forceinline__ float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

// Kernel that computes a partial product over a tile of the K-dimension and uses atomicAdd to accumulate results
__global__ void matmul_atomic_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K,
                                       int lda, int ldb, int ldc,
                                       bool transA, bool transB) {
    // Compute the row and column for this thread (tile in MxN)
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Determine the starting index in the K dimension for this block
    int k_tile = blockIdx.z * TILE_K;

    float sum = 0.0f;

    // Declare shared memory for tiles from A and B
    __shared__ float As[BLOCK_SIZE][TILE_K];
    __shared__ float Bs[TILE_K][BLOCK_SIZE];

    // Load a tile of matrix A into shared memory
    if (row < M && (k_tile + threadIdx.x) < K) {
        As[threadIdx.y][threadIdx.x] = get_element(A, row, k_tile + threadIdx.x, lda, transA);
    } else {
        As[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Load a tile of matrix B into shared memory
    if (col < N && (k_tile + threadIdx.y) < K) {
        Bs[threadIdx.y][threadIdx.x] = get_element(B, k_tile + threadIdx.y, col, ldb, transB);
    } else {
        Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Compute the partial dot product over the current K-tile
    #pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    // Use atomicAdd to accumulate the computed partial sum into global memory
    if (row < M && col < N) {
        atomicAdd(&C[row * ldc + col], sum);
    }
}

// Host function interfacing with PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Validate that tensors are CUDA tensors and are 2D
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Retrieve input dimensions
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int64_t M, N, K;
    int lda, ldb, ldc;

    // Determine dimensions and transpose flags based on input shapes
    if (A_rows >= A_cols && B_rows == A_cols) {
        // A (M x K), B (K x N)
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

    // Allocate output tensor with zeros, since we use atomicAdd for accumulation
    auto C = torch::zeros({M, N}, A.options());

    // Configure a 3D grid: (grid_x, grid_y, grid_z) where grid_z partitions the K dimension
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (K + TILE_K - 1) / TILE_K);

    // Launch the kernel
    matmul_atomic_kernel<<<gridDim, blockDim>>>(
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
    m.def("forward", &matmul_cuda, "Matrix multiplication using atomic accumulation for tall and skinny matrices (CUDA)");
}
