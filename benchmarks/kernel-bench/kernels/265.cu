#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for a single matrix multiplication
__device__ float matrix_multiply_element(const float* __restrict__ A, const float* __restrict__ B, int M, int K, int N, int m, int n) {
    float val = 0.0f;
    for (int k = 0; k < K; k++) {
        val += A[m * K + k] * B[k * N + n];
    }
    return val;
}

// CUDA kernel for batched matrix multiplication: C = A * B
// Shapes: A (batch_size, M, K), B (batch_size, K, N), C (batch_size, M, N)
__global__ void bmm_kernel_modular(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * M * N;
    if (idx >= total) return;

    int b = idx / (M * N);
    int remainder = idx % (M * N);
    int m = remainder / N;
    int n = remainder % N;

    C[b * M * N + m * N + n] = matrix_multiply_element(&A[b * M * K], &B[b * K * N], M, K, N, m, n);
}

torch::Tensor forward_bmm_modular(torch::Tensor A, torch::Tensor B) {
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

    int total = batch_size * M * N;
    const int threads = 256;
    int blocks = (total + threads - 1) / threads;

    bmm_kernel_modular<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm_modular, "Batched matrix multiplication (CUDA)");
}