#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void bmm_kernel(float* shared_A, float* shared_B, const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int batch_size, int M, int K, int N) {
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Batch index
    const int bz = blockIdx.z;
    // Compute global row and column index for the current thread
    const int row_start = blockIdx.y * blockDim.y + threadIdx.y;
    const int col_start = blockIdx.x * blockDim.x + threadIdx.x;

    // Use stride to ensure covering larger M or N than available threads
    for (int row = row_start; row < M; row += blockDim.y * gridDim.y) {
        for (int col = col_start; col < N; col += blockDim.x * gridDim.x) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[bz * M * K + row * K + k] *
                       B[bz * K * N + k * N + col];
            }
            C[bz * M * N + row * N + col] = sum;
        }
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(
        (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (M + BLOCK_SIZE - 1) / BLOCK_SIZE,
        batch_size
    );

    bmm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication (CUDA)");
}
