#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Static cuBLAS handle to avoid recreation costs
static cublasHandle_t handle = nullptr;

// Kernel: Tiled matrix multiplication with register tiling
// Each block computes a BLOCK_SIZE x BLOCK_SIZE tile of C, and each thread computes a 2x2 sub-tile.
// Shared memory is used to load tiles of A and B, while register tiling improves arithmetic intensity.
__global__ void matmul_kernel_tiled_regtile(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K) {
    // Determine the starting indices of the C tile for this block
    int row_start = blockIdx.y * BLOCK_SIZE;
    int col_start = blockIdx.x * BLOCK_SIZE;

    // Each block is launched with dimensions:
    // blockDim.x = BLOCK_SIZE / THREAD_TILE_N, blockDim.y = BLOCK_SIZE / THREAD_TILE_M
    // (e.g., 16 x 16 = 256 threads per block)
    int tx = threadIdx.x; // horizontal index in the sub-tile grid
    int ty = threadIdx.y; // vertical index in the sub-tile grid

    // Compute the top-left position of the sub-tile computed by this thread within the block tile
    int sub_row = ty * THREAD_TILE_M;  // starting row offset within the block tile
    int sub_col = tx * THREAD_TILE_N;  // starting col offset within the block tile

    // Registers for the output sub-tile (THREAD_TILE_M x THREAD_TILE_N per thread)
    float accum[THREAD_TILE_M][THREAD_TILE_N] = {0.0f};

    // Shared memory tiles for A and B
    // A tile: BLOCK_SIZE rows x TILE_K columns
    // B tile: TILE_K rows x BLOCK_SIZE columns
    __shared__ float sharedA[BLOCK_SIZE][TILE_K];
    __shared__ float sharedB[TILE_K][BLOCK_SIZE];

    // Loop over tiles of K dimension
    int numTiles = (K + TILE_K - 1) / TILE_K;
    for (int t = 0; t < numTiles; t++) {
        // Cooperative loading of A tile into shared memory
        // Total elements in A tile: BLOCK_SIZE * TILE_K
        int threadId = threadIdx.y * blockDim.x + threadIdx.x; // linear thread index in the block
        int totalA = BLOCK_SIZE * TILE_K;
        for (int i = threadId; i < totalA; i += blockDim.x * blockDim.y) {
            int r = i / TILE_K;  // row index within the A tile
            int c = i % TILE_K;  // col index within the A tile
            int globalRow = row_start + r;
            int globalCol = t * TILE_K + c;
            if (globalRow < M && globalCol < K)
                sharedA[r][c] = A[globalRow * K + globalCol];
            else
                sharedA[r][c] = 0.0f;
        }

        // Cooperative loading of B tile into shared memory
        // Total elements in B tile: TILE_K * BLOCK_SIZE
        int totalB = TILE_K * BLOCK_SIZE;
        for (int i = threadId; i < totalB; i += blockDim.x * blockDim.y) {
            int r = i / BLOCK_SIZE;  // row index within the B tile
            int c = i % BLOCK_SIZE;  // col index within the B tile
            int globalRow = t * TILE_K + r;
            int globalCol = col_start + c;
            if (globalRow < K && globalCol < N)
                sharedB[r][c] = B[globalRow * N + globalCol];
            else
                sharedB[r][c] = 0.0f;
        }

        __syncthreads(); // Ensure the tiles are fully loaded

        // Compute the partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            // Load a vector of values from sharedA corresponding to the sub-tile rows
            float a_vals[THREAD_TILE_M];
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                a_vals[i] = sharedA[sub_row + i][k];
            }
            // Load a vector of values from sharedB corresponding to the sub-tile columns
            float b_vals[THREAD_TILE_N];
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_N; j++) {
                b_vals[j] = sharedB[k][sub_col + j];
            }
            // Multiply-accumulate the sub-tile elements
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    accum[i][j] += a_vals[i] * b_vals[j];
                }
            }
        }
        __syncthreads(); // Prepare for the next tile load
    }

    // Write the computed sub-tile results back to global memory
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

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    float *d_A = A.data_ptr<float>();
    float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    // Initialize handle only once
    if (handle == nullptr) {
        cublasCreate(&handle);
        // Set math mode to handle Tensor Cores if available
        cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use CUBLAS_OP_N for both matrices since they are already in the correct layout
    // Perform matrix multiplication C = A * B
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N,    // leading dimension of B
                d_A, K,    // leading dimension of A
                &beta,
                d_C, N);   // leading dimension of C

    // Launch custom kernel for additional optimization
    dim3 blockDim(BLOCK_SIZE / THREAD_TILE_N, BLOCK_SIZE / THREAD_TILE_M);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matmul_kernel_tiled_regtile<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int N = B.size(1);

    // Create output tensor with same options as input
    auto options = torch::TensorOptions()
        .dtype(A.dtype())
        .device(A.device())
        .requires_grad(false);
    
    torch::Tensor C = torch::empty({M, N}, options);

    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid matrix multiplication using cuBLAS and custom kernel (CUDA)");
}