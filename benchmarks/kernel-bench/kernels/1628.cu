#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

__constant__ int d_N;  // Matrix dimension in constant memory

__global__ void upper_triangular_matmul_kernel(const float* __restrict__ A, 
                                             const float* __restrict__ B, 
                                             float* __restrict__ C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < d_N && col < d_N && row <= col) {
        float sum = 0.0f; 
    #pragma unroll 4
        #pragma unroll 4
        for (int k = row; k <= col; ++k) {
            sum += A[row * d_N + k] * B[k * d_N + col];
        }
        C[row * d_N + col] = sum;
    }
}

torch::Tensor upper_triangular_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    // Copy matrix dimension to constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));

    dim3 threadsPerBlock(32, 16);  // Maintain warp alignment
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    upper_triangular_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>()
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul, "Upper triangular matrix multiplication with constant memory");
}