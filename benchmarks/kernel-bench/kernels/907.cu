#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// CUDA kernel using grid-stride loops to handle irregular shapes
__global__ void matmul_stride_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // Compute initial indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Use grid-stride loops to cover all rows and columns
    for (int i = row; i < M; i += blockDim.y * gridDim.y) {
        for (int j = col; j < N; j += blockDim.x * gridDim.x) {
            float sum = 0.0f;
            // Compute dot product for C[i,j]
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Host function called from PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Check that inputs are CUDA tensors and contiguous
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create an output tensor C of shape (M, N)
    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Define thread block dimensions
    dim3 block(16, 16);
    // Calculate grid dimensions to cover entire output matrix
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    // Launch the kernel using grid-stride loops for boundary correctness
    matmul_stride_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    // Error checking and synchronization
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return C;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Matrix multiplication with stride loops (CUDA)");
}
