#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 16

// Branchless safe load for matrix A: clamps indices to always read valid memory and multiplies by a mask computed without divergent branches
__device__ inline float safeLoadA(const float *A, int row, int col, int M, int K) {
    int safe_row = row < M ? row : M - 1;
    int safe_col = col < K ? col : K - 1;
    // Compute mask: 1.0 if within bounds, 0.0 otherwise
    float mask = (row < M && col < K) ? 1.0f : 0.0f;
    return A[safe_row * K + safe_col] * mask;
}

// Branchless safe load for matrix B: clamps indices to always read valid memory and multiplies by a mask
__device__ inline float safeLoadB(const float *B, int row, int col, int K, int N) {
    int safe_row = row < K ? row : K - 1;
    int safe_col = col < N ? col : N - 1;
    float mask = (row < K && col < N) ? 1.0f : 0.0f;
    return B[safe_row * N + safe_col] * mask;
}

__global__ void matmul_branchless_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float sum = 0.0f;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Loop over tiles in the K dimension
    for (int t = 0; t < K; t += TILE_SIZE) {
        // Use branchless safe loads to avoid divergent conditions
        tileA[ty][tx] = safeLoadA(A, row, t + tx, M, K);
        tileB[ty][tx] = safeLoadB(B, t + ty, col, K, N);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum = __fmaf_rn(tileA[ty][k], tileB[k][tx], sum);
        }
        __syncthreads();
    }

    // Write the result if within valid bounds
    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_branchless_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Branchless Matrix Multiplication (CUDA)");
}
