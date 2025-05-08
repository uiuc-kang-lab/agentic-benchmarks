#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

__global__ void matMulKernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int K,
                             int M,
                             int N) {
    __shared__ float s_A[16][16];
    __shared__ float s_B[16][16];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (K + 15) / 16; ++tile) {
        // Load data into shared memory
        if (row < M && (tile * 16 + tx) < K)
            s_A[ty][tx] = A[(tile * 16 + tx) * M + row];
        else
            s_A[ty][tx] = 0.0f;

        if (col < N && (tile * 16 + ty) < K)
            s_B[ty][tx] = B[(tile * 16 + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        // Compute partial dot product
        if (row < M && col < N) {
            for (int k = 0; k < 16 && (tile * 16 + k) < K; ++k) {
                sum += s_A[ty][k] * s_B[k][tx];
            }
        }
        
        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");

    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(16, 16);
    dim3 grid((M + 15)/16, (N + 15)/16);

    matMulKernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matmul with transposed A");
}