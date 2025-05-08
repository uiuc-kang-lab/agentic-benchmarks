#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_FACTOR 2
#define TILE_DIM (BLOCK_SIZE * TILE_FACTOR)  // Each block computes a TILE_DIM x TILE_DIM output tile

// Helper to fetch matrix elements considering transpose
__device__ inline float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

// Kernel: each thread computes a 2x2 block of output, reducing the total thread count and aligning work
__global__ void matmul_kernel_multitile(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K,
                                         int lda, int ldb, int ldc,
                                         bool transA, bool transB) {
    // Determine the starting row and column for the block
    int blockRow = blockIdx.y * TILE_DIM;
    int blockCol = blockIdx.x * TILE_DIM;

    // Each thread computes a 2x2 tile within the block
    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int row0 = blockRow + threadRow * TILE_FACTOR;  // starting row for this thread's tile
    int col0 = blockCol + threadCol * TILE_FACTOR;  // starting col for this thread's tile

    // Accumulators for a 2x2 output computed in registers
    float acc[2][2] = { {0.0f, 0.0f}, {0.0f, 0.0f} };

    // Shared memory tiles
    __shared__ float As[TILE_DIM][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][TILE_DIM];

    // Loop over tiles on the K dimension
    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        int tiledK = t * BLOCK_SIZE;
        
        // Load a tile of matrix A into shared memory
        // Each thread loads TILE_FACTOR elements from A
        for (int i = 0; i < TILE_FACTOR; ++i) {
            int globalRow = row0 + i;
            int globalCol = tiledK + threadIdx.x;
            if (globalRow < M && globalCol < K)
                As[threadIdx.y * TILE_FACTOR + i][threadIdx.x] = get_element(A, globalRow, globalCol, lda, transA);
            else
                As[threadIdx.y * TILE_FACTOR + i][threadIdx.x] = 0.0f;
        }

        // Load a tile of matrix B into shared memory
        // Each thread loads TILE_FACTOR elements from B
        for (int i = 0; i < TILE_FACTOR; ++i) {
            int globalRow = tiledK + threadIdx.y;
            int globalCol = col0 + i;
            if (globalRow < K && globalCol < N)
                Bs[threadIdx.y][threadIdx.x * TILE_FACTOR + i] = get_element(B, globalRow, globalCol, ldb, transB);
            else
                Bs[threadIdx.y][threadIdx.x * TILE_FACTOR + i] = 0.0f;
        }
        
        __syncthreads();
        
        // Multiply the loaded tiles
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a0 = As[threadIdx.y * TILE_FACTOR + 0][k];
            float a1 = As[threadIdx.y * TILE_FACTOR + 1][k];
            float b0 = Bs[k][threadIdx.x * TILE_FACTOR + 0];
            float b1 = Bs[k][threadIdx.x * TILE_FACTOR + 1];
            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
        }
        
        __syncthreads();
    }

    // Write back the computed 2x2 tile to global memory
    for (int i = 0; i < TILE_FACTOR; ++i) {
        for (int j = 0; j < TILE_FACTOR; ++j) {
            int globalRow = row0 + i;
            int globalCol = col0 + j;
            if (globalRow < M && globalCol < N)
                C[globalRow * ldc + globalCol] = acc[i][j];
        }
    }
}


torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    bool transA = false;
    bool transB = false;
    int64_t M, N, K;
    int lda, ldb, ldc;

    if (A_rows >= A_cols && B_rows == A_cols) {
        // A is M x K, B is K x N
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        // A is transposed: K x M, so treat it as M x K with transA=true
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        // B is transposed: N x K, so treat it as K x N with transB=true
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
        // Both A and B are transposed
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
    auto C = torch::empty({M, N}, A.options());

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matmul_kernel_multitile<<<gridDim, blockDim>>>(
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
    m.def("forward", &matmul_cuda, "Matrix multiplication with optimized thread/block indexing (CUDA)");
}
