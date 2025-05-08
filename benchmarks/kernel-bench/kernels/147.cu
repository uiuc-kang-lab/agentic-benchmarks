#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel to perform matrix multiplication
__global__ void matrixMultiplyKernel(float *A, float *B, float *C, int M, int N, int K) {
    __shared__ float shared_A[32][32];
    __shared__ float shared_B[32][32];
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float value = 0;
    for (int e = 0; e < (K + blockDim.x - 1) / blockDim.x; ++e) {
        if (e * blockDim.x + threadIdx.x < K && row < M) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + e * blockDim.x + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (e * blockDim.x + threadIdx.y < K && col < N) {
            shared_B[threadIdx.y][threadIdx.x] = B[(e * blockDim.x + threadIdx.y) * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();
        for (int k = 0; k < blockDim.x; ++k) {
            value += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float value = 0;
        for (int e = 0; e < K; ++e) {
            value += A[row * K + e] * B[e * N + col];
        }
        C[row * N + col] = value;
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    // Ensure inputs are CUDA tensors and contiguous
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    // Get the dimensions of the matrices
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Get the pointers to the data
    float *d_A = A.data_ptr<float>();
    float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    // Define block size and grid size
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors and contiguous
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    // Get the dimensions of the matrices
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create the output tensor
    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Perform the matrix multiplication
    matrix_multiply_cuda(A, B, C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication (CUDA)");
}