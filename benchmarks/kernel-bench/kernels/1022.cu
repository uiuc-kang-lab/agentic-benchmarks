#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define TILE_SIZE 16

// Device function to load a tile from matrix A
__device__ void loadTileA(float (&As)[TILE_SIZE][TILE_SIZE],
                         const float* __restrict__ A,
                         int row, int aCol,
                         int M, int K,
                         int lda, bool transA) {
    if (row < M && aCol < K) {
        As[threadIdx.y][threadIdx.x] = transA ? 
            __ldg(&A[aCol * lda + row]) : 
            __ldg(&A[row * lda + aCol]);
    } else {
        As[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

// Device function to load a tile from matrix B
__device__ void loadTileB(float (&Bs)[TILE_SIZE][TILE_SIZE],
                         const float* __restrict__ B,
                         int bRow, int col,
                         int K, int N,
                         int ldb, bool transB) {
    if (bRow < K && col < N) {
        Bs[threadIdx.y][threadIdx.x] = transB ? 
            __ldg(&B[col * ldb + bRow]) : 
            __ldg(&B[bRow * ldb + col]);
    } else {
        Bs[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

// Device function to compute partial product for a tile
__device__ float computeTileProduct(const float (&As)[TILE_SIZE][TILE_SIZE],
                                  const float (&Bs)[TILE_SIZE][TILE_SIZE]) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k += 4) {
        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        sum += As[threadIdx.y][k+1] * Bs[k+1][threadIdx.x];
        sum += As[threadIdx.y][k+2] * Bs[k+2][threadIdx.x];
        sum += As[threadIdx.y][k+3] * Bs[k+3][threadIdx.x];
    }
    return sum;
}

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int N, int K,
                             int lda, int ldb, int ldc,
                             bool transA, bool transB) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // Load tiles using modular functions
        loadTileA(As, A, row, aCol, M, K, lda, transA);
        loadTileB(Bs, B, bRow, col, K, N, ldb, transB);

        __syncthreads();

        // Compute partial product using modular function
        sum += computeTileProduct(As, Bs);

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * ldc + col] = sum;
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
    int M, N, K;
    int lda, ldb, ldc;

    if (A_rows >= A_cols && B_rows == A_cols) {
        M = A_rows;
        K = A_cols;
        N = B_cols;
        lda = A.stride(0);
        ldb = B.stride(0);
    } else if (A_cols > A_rows && B_rows == A_rows) {
        transA = true;
        M = A_cols;
        K = A_rows;
        N = B_cols;
        lda = A.stride(1);
        ldb = B.stride(0);
    } else if (A_rows >= A_cols && B_cols == A_cols) {
        transB = true;
        M = A_rows;
        K = A_cols;
        N = B_rows;
        lda = A.stride(0);
        ldb = B.stride(1);
    } else if (A_cols > A_rows && B_cols == A_rows) {
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Modular Tiled Matrix Multiplication (CUDA)");
}