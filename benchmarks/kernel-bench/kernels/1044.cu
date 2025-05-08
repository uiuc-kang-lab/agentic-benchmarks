#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use a tile size that is a multiple of the warp size to improve utilization
#define TILE_SIZE 32

// Device function to fetch an element from a matrix with optional transpose
__device__ inline float get_val(const float* __restrict__ mat, int row, int col, int ld, bool transpose) {
    return transpose ? mat[col * ld + row] : mat[row * ld + col];
}

// Kernel using improved 2D thread and block indexing
__global__ void optimized_matmul_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K,
                                          int lda, int ldb, int ldc,
                                          bool transA, bool transB) {
    // Allocate shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Calculate global row and column indices for C directly using 2D mapping
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // Global row index
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // Global col index

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over all tiles
    for (int t = 0; t < numTiles; t++) {
        int tiledA_col = t * TILE_SIZE + threadIdx.x;
        int tiledB_row = t * TILE_SIZE + threadIdx.y;

        // Load A tile: each thread loads one element if within bounds
        if (row < M && tiledA_col < K)
            As[threadIdx.y][threadIdx.x] = get_val(A, row, tiledA_col, lda, transA);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile: each thread loads one element if within bounds
        if (col < N && tiledB_row < K)
            Bs[threadIdx.y][threadIdx.x] = get_val(B, tiledB_row, col, ldb, transB);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads(); // Ensure the tile is loaded

        // Compute partial product for the tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write result if within output bounds
    if (row < M && col < N)
        C[row * ldc + col] = sum;
}


// Host function to infer dimensions and launch the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check inputs
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Extract dimensions
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    // Determine multiplication mode and possible transpositions
    bool transA = false, transB = false;
    int64_t M, N, K;
    int lda, ldb, ldc;

    if (A_rows >= A_cols && B_rows == A_cols) {
        // A: M x K, B: K x N
        M = A_rows; K = A_cols; N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A stored transposed
        transA = true;
        M = A_cols; K = A_rows; N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B stored transposed
        transB = true;
        M = A_rows; K = A_cols; N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Both A and B stored transposed
        transA = true; transB = true;
        M = A_cols; K = A_rows; N = B_rows;
        lda = A.stride(1);
        ldb = B.stride(1);
    } else {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }
    ldc = N;

    // Allocate output tensor
    auto C = torch::empty({M, N}, A.options());

    // Configure a 2D grid and block layout using TILE_SIZE x TILE_SIZE threads per block
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the optimized kernel
    optimized_matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB
    );
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Optimized matrix multiplication with improved thread and block indexing (CUDA)");
}
