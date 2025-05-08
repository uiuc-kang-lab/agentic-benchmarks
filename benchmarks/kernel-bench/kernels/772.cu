#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 16

__device__ inline float safeLoad(const float *matrix, int row, int col, int max_row, int max_col, int ld) {
    int safe_row = min(row, max_row - 1);
    int safe_col = min(col, max_col - 1);
    float mask = (row < max_row && col < max_col) ? 1.0f : 0.0f;
    return matrix[safe_row * ld + safe_col] * mask;
}

__global__ void matmul_optimized_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float sum = 0.0f;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    for (int t = 0; t < K; t += TILE_SIZE) {
        tileA[ty][tx] = safeLoad(A, row, t + tx, M, K, K);
        tileB[ty][tx] = safeLoad(B, t + ty, col, K, N, N);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum = __fmaf_rn(tileA[ty][k], tileB[k][tx], sum);
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_optimized_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Branchless Matrix Multiplication (CUDA)");
}