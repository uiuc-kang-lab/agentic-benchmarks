#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Tile dimensions for custom kernel
#define BLOCK_SIZE 32     // Overall tile size for C (BLOCK_SIZE x BLOCK_SIZE)
#define TILE_K 32         // Tile width for the K dimension

// Register tiling dimensions: each thread computes a small sub-tile
#define THREAD_TILE_M 2   // Each thread computes 2 rows
#define THREAD_TILE_N 2   // Each thread computes 2 columns

// Macros for input checks
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);   \
    CHECK_CONTIGUOUS(x)

// Custom tiled matrix multiplication kernel with register tiling
// Computes a BLOCK_SIZE x BLOCK_SIZE tile of C per block, where each thread computes a 2x2 sub-tile
__global__ void matmul_kernel_tiled_regtile(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K) {
    // Determine the origin of the tile in C
    int row_start = blockIdx.y * BLOCK_SIZE;
    int col_start = blockIdx.x * BLOCK_SIZE;

    // Indices in the sub-tile computed by this thread
    int tx = threadIdx.x; // horizontal index
    int ty = threadIdx.y; // vertical index
    int sub_row = ty * THREAD_TILE_M;
    int sub_col = tx * THREAD_TILE_N;

    // Accumulator for the sub-tile computed in registers
    float accum[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    // Shared memory tiles for A and B
    __shared__ float sharedA[BLOCK_SIZE][TILE_K];
    __shared__ float sharedB[TILE_K][BLOCK_SIZE];

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_K - 1) / TILE_K;
    for (int t = 0; t < numTiles; t++) {
        // Each thread cooperatively loads parts of A into shared memory
        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        int totalA = BLOCK_SIZE * TILE_K;
        for (int i = threadId; i < totalA; i += blockDim.x * blockDim.y) {
            int r = i / TILE_K;
            int c = i % TILE_K;
            int globalRow = row_start + r;
            int globalCol = t * TILE_K + c;
            sharedA[r][c] = (globalRow < M && globalCol < K) ? A[globalRow * K + globalCol] : 0.0f;
        }

        // Each thread cooperatively loads parts of B into shared memory
        int totalB = TILE_K * BLOCK_SIZE;
        for (int i = threadId; i < totalB; i += blockDim.x * blockDim.y) {
            int r = i / BLOCK_SIZE;
            int c = i % BLOCK_SIZE;
            int globalRow = t * TILE_K + r;
            int globalCol = col_start + c;
            sharedB[r][c] = (globalRow < K && globalCol < N) ? B[globalRow * N + globalCol] : 0.0f;
        }

        __syncthreads(); // Ensure shared memory loads are complete

        // Compute partial products for the tile
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
        __syncthreads();
    }

    // Write results from registers back to global memory
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int globalRow = row_start + sub_row + i;
            int globalCol = col_start + sub_col + j;
            if (globalRow < M && globalCol < N)
                C[globalRow * N + globalCol] = accum[i][j];
        }
    }
}

// Hybrid host function: selects the custom kernel for small matrices, and uses cuBLAS for larger ones
void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float *d_A = A.data_ptr<float>();
    const float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    // Use the custom tiled kernel for smaller matrix sizes; otherwise, use cuBLAS
    if (M <= 256 && N <= 256 && K <= 256) {
        // Launch parameters for the custom kernel
        dim3 blockDim(BLOCK_SIZE / THREAD_TILE_N, BLOCK_SIZE / THREAD_TILE_M);
        dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
        matmul_kernel_tiled_regtile<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    } else {
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        // Note: cuBLAS expects column-major order. Here, we use a transposition trick to match our row-major layout
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
}

// PyTorch binding
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int N = B.size(1);
    auto options = torch::TensorOptions()
        .dtype(A.dtype())
        .device(A.device())
        .requires_grad(false);
    torch::Tensor C = torch::empty({M, N}, options);

    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid matrix multiplication (custom tiled kernel for small matrices, cuBLAS for large ones)");
}
