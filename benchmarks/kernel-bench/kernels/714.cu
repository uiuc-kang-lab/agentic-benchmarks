#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using grid-stride loops to cover the entire output matrix
__global__ void StrideMatMulKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int N, int K) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    // Each thread handles multiple elements via stride loops
    for (int index = tid; index < total; index += gridDim.x * blockDim.x) {
        int row = index / N;
        int col = index % N;
        float sum = 0.0f;
        // Compute dot product for element C[row, col]
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Host function exposed via PyBind11
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());
    
    // Flatten the output matrix dimension for mapping threads
    int total = M * N;
    int blockSize = 256;
    int numBlocks = (total + blockSize - 1) / blockSize;

    StrideMatMulKernel<<<numBlocks, blockSize>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided grid loop matrix multiplication (CUDA)");
}
