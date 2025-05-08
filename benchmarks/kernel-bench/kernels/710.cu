#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 8

__global__ void MatrixMulKernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               const int M, const int N, const int K) {
    // Calculate global thread indices
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE_Y + ty;
    const int col = blockIdx.x * BLOCK_SIZE_X + tx;

    // Compute only if within bounds
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Since K is small, we process it sequentially
        #pragma unroll 4
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Write result
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Configure thread blocks and grid
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);
    dim3 numBlocks(
        (N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
        (M + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y
    );

    // Launch kernel
    MatrixMulKernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized thread mapping matrix multiplication (CUDA)");
}