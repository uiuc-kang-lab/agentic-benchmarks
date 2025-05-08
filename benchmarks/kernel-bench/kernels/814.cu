#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel: each thread computes one element of C using direct dot product (suitable for small K)
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              const int M, const int K, const int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (tid < total) {
        int row = tid / N;
        int col = tid % N;
        float sum = 0.0f;
        #pragma unroll
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[tid] = sum;
    }
}

// The forward function wraps the kernel launch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Tensor B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());
    int total = M * N;
    int blockSize = 1024;
    int gridSize = (total + blockSize - 1) / blockSize;

    matmul_kernel<<<gridSize, blockSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Small K matrix multiplication kernel using 1D thread mapping (CUDA)");
}
