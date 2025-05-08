#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel using grid-stride loops to handle large workloads
__global__ void matrixMulSharedKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    extern __shared__ float shared[];
    float* As = shared;
    float* Bs = shared + blockDim.x * blockDim.y;

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    for (int tileIdx = 0; tileIdx < (K + blockDim.x - 1) / blockDim.x; ++tileIdx) {
        if (row < M && tileIdx * blockDim.x + threadIdx.x < K) {
            As[threadIdx.y * blockDim.x + threadIdx.x] = A[row * K + tileIdx * blockDim.x + threadIdx.x];
        } else {
            As[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
        }

        if (col < N && tileIdx * blockDim.y + threadIdx.y < K) {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = B[(tileIdx * blockDim.y + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y * blockDim.x + threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            sum += As[threadIdx.y * blockDim.x + k] * Bs[k * blockDim.x + threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Function to launch the CUDA kernel
void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    int totalElements = M * N;
    int threads = 256;  // defining block size
    int blocks = (totalElements + threads - 1) / threads;  // grid size computed based on total workload

    matrixMulStrideKernel<<<blocks, threads>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
}

// Pybind11 interface: creates an output tensor, calls the CUDA kernel, returns the result
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create output tensor C and initialize to zero
    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with grid-stride loops (CUDA)");
}
