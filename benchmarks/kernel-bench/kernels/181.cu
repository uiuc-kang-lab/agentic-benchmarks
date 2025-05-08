#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel using grid-stride loops to handle large workloads
__global__ void matrixMulStrideKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N; // total number of elements in output matrix C
    int gridSize = blockDim.x * gridDim.x;

    // Each thread works over multiple output elements using stride looping
    for (int index = idx; index < total; index += gridSize) {
        int row = index / N;
        int col = index % N;
        float sum = 0.0f;
        
        // Compute dot product for element C(row, col)
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
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
