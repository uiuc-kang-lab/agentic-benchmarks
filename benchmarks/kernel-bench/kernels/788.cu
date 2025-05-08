#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel: each thread computes one element of the output matrix C = A * B.
// Using 2D block and grid indexing ensures that threads are mapped efficiently
// to the problem domain, which is particularly beneficial for a small K dimension.

__global__ void matmul_kernel(const float* __restrict__ A, 
                               const float* __restrict__ B, 
                               float* __restrict__ C,
                               int M, int K, int N) {
    // Compute the row and column index of the element
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        // Loop over the small K dimension
        #pragma unroll
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// The forward function wraps the kernel launch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor C
    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Define tile dimensions for the 2D thread block
    const int TILE_DIM = 16; // Using 16x16 thread blocks
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch the kernel with 2D grid and block indexing
    matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    // Check for any kernel launch errors
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed : ", cudaGetErrorString(err));

    return C;
}

// Bind the forward function to the pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with 2D thread indexing (CUDA)");
}
