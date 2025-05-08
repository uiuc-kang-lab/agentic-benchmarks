#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for batched matrix multiplication using stride loops: C = A * B
// Shapes: A (batch_size, M, K), B (batch_size, K, N), C (batch_size, M, N)
__global__ void bmm_kernel_stride(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int total = batch_size * M * N;

    for (int i = idx; i < total; i += stride) {
        int b = i / (M * N);
        int remainder = i % (M * N);
        int m = remainder / N;
        int n = remainder % N;

        float val = 0.0f;
        for (int k = 0; k < K; k++) {
            val += A[b * M * K + m * K + k] * B[b * K * N + k * N + n];
        }
        C[b * M * N + m * N + n] = val;
    }
}

torch::Tensor forward_bmm_stride(torch::Tensor A, torch::Tensor B) {
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

    bmm_kernel_stride<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm_stride, "Batched matrix multiplication with stride (CUDA)");
}