#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define MAX_DIAG_SIZE 16384

__constant__ float d_A[MAX_DIAG_SIZE];

__global__ void diag_matmul_stride_kernel(
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    const int64_t total = N * M;
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = blockDim.x * gridDim.x;
    
    while (idx < total) {
        const int64_t row = idx / M;
        C[idx] = d_A[row] * B[idx];
        idx += stride;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch");
    TORCH_CHECK(A.size(0) <= MAX_DIAG_SIZE, "Diagonal matrix too large");

    A = A.contiguous();
    B = B.contiguous();

    const int64_t N = A.size(0);
    const int64_t M = B.size(1);

    cudaMemcpyToSymbol(d_A, A.data_ptr<float>(), N * sizeof(float));

    auto C = torch::empty({N, M}, B.options());

    const int threads = 256;
    const int blocks = (N * M + threads - 1) / threads;
    diag_matmul_stride_kernel<<<blocks, threads>>>(
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matmul with stride optimization");
}