#include <torch/extension.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 4  // Reduced tile size for warp-level operations

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int K, int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;

    if (row < M && col < N) {
        // Process multiple elements per thread
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }

        // Warp-level reduction using shfl_down
        for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        // First thread in warp writes result
        if (threadIdx.x % WARP_SIZE == 0) {
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor"); TORCH_CHECK(A.is_contiguous(), "Input A must be contiguous"); TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor"); TORCH_CHECK(B.is_contiguous(), "Input B must be contiguous");
    CHECK_CUDA(B); CHECK_CONTIGUOUS(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    dim3 threadsPerBlock(WARP_SIZE, TILE_SIZE);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-reduced matrix multiplication (CUDA)");
}