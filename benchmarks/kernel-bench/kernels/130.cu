/*
Hybrid Register-Tiled Matrix Multiplication
This implementation combines a highly-optimized custom CUDA kernel using register tiling for small matrix multiplications and falls back on cuBLAS for larger matrices. The custom kernel is based on a 32x32 block tiling strategy with additional register tiling (2x2 per thread), which improves arithmetic intensity. For larger matrices, cuBLAS is invoked to take advantage of optimized vendor libraries and potentially GPU tensor cores.

Compile as a PyTorch CUDA extension.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Tiling configurations
#define BLOCK_SIZE 32     // Overall tile size for C (BLOCK_SIZE x BLOCK_SIZE)
#define TILE_K 32         // Tile width along the K dimension
#define THREAD_TILE_M 2   // Each thread computes 2 rows
#define THREAD_TILE_N 2   // Each thread computes 2 columns

// Heuristic threshold: for matrices smaller than or equal to this dimension, use the custom kernel
#define SMALL_MATRIX_DIM 128

// Macros for input validations
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Static cuBLAS handle to avoid repeated creation overhead
static cublasHandle_t cublas_handle = nullptr;

// Custom CUDA kernel: Tiled matrix multiplication with register tiling
__global__ void matmul_kernel_tiled_regtile(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K) {
    // Identify the tile for this block
    int row_start = blockIdx.y * BLOCK_SIZE;
    int col_start = blockIdx.x * BLOCK_SIZE;

    // Each thread computes a small sub-tile (THREAD_TILE_M x THREAD_TILE_N)
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int sub_row = ty * THREAD_TILE_M;
    int sub_col = tx * THREAD_TILE_N;

    // Registers to accumulate C sub-tile results
    float accum[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    // Shared memory tiles for A and B
    __shared__ float sharedA[BLOCK_SIZE][TILE_K];
    __shared__ float sharedB[TILE_K][BLOCK_SIZE];

    // Loop over tiles along the K dimension
    int numTiles = (K + TILE_K - 1) / TILE_K;
    for (int t = 0; t < numTiles; t++) {
        // Cooperative loading of A tile into shared memory
        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        int totalA = BLOCK_SIZE * TILE_K;
        for (int i = threadId; i < totalA; i += blockDim.x * blockDim.y) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int globalRow = row_start + r;
            int globalCol = t * TILE_K + c;
            if (globalRow < M && globalCol < K)
                sharedA[r][c] = A[globalRow * K + globalCol];
            else
                sharedA[r][c] = 0.0f;
        }

        // Cooperative loading of B tile into shared memory
        int totalB = TILE_K * BLOCK_SIZE;
        for (int i = threadId; i < totalB; i += blockDim.x * blockDim.y) {
            int r = i / BLOCK_SIZE;
            int c = i % BLOCK_SIZE;
            int globalRow = t * TILE_K + r;
            int globalCol = col_start + c;
            if (globalRow < K && globalCol < N)
                sharedB[r][c] = B[globalRow * N + globalCol];
            else
                sharedB[r][c] = 0.0f;
        }

        __syncthreads(); // Ensure shared tiles are loaded

        // Compute the partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            float a_vals[THREAD_TILE_M];
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                a_vals[i] = sharedA[sub_row + i][k];
            }
            float b_vals[THREAD_TILE_N];
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {
                b_vals[j] = sharedB[k][sub_col + j];
            }
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    accum[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }
        __syncthreads(); // Synchronize before loading the next tile
    }

    // Write the computed sub-tile back to global memory
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int globalRow = row_start + sub_row + i;
            int globalCol = col_start + sub_col + j;
            if (globalRow < M && globalCol < N) {
                C[globalRow * N + globalCol] = accum[i][j];
            }
        }
    }
}

// Host function that selects between the custom kernel and cuBLAS based on problem size
void matrix_multiply_cuda(const torch::Tensor &A,
                            const torch::Tensor &B,
                            torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // Use the custom register-tiled kernel for small matrices to reduce launch overhead
    if (M <= SMALL_MATRIX_DIM && N <= SMALL_MATRIX_DIM && K <= SMALL_MATRIX_DIM) {
        // Define block dimensions matching the register tiling strategy
        dim3 blockDim(BLOCK_SIZE / THREAD_TILE_N, BLOCK_SIZE / THREAD_TILE_M);
        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                     (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_kernel_tiled_regtile<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    } else {
        // For larger matrices, use the cuBLAS library
        if (cublas_handle == nullptr) {
            cublasCreate(&cublas_handle);
            cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // cuBLAS assumes column-major storage. To process row-major matrices correctly,
        // we swap A and B in the call.
        cublasSgemm(cublas_handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
}

// PyTorch forward interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    int M = A.size(0);
    int N = B.size(1);
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device()).requires_grad(false);
    torch::Tensor C = torch::empty({M, N}, options);
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid register-tiled matrix multiplication (CUDA): custom kernel for small matrices and cuBLAS for larger ones");
}
