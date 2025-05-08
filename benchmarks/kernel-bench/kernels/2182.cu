#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Kernel for A.T * B utilizing warp-level reduction with __shfl_down_sync
__global__ void warpOptimizedReductionKernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int K,
                               int M, 
                               int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float sum = 0.0f;
        int lane = threadIdx.x % 32;  // Warp lane index
        for (int k = 0; k < K; k += 32) {
            float partialSum = 0.0f;
            if (k + lane < K)
                partialSum = A[(k + lane) * M + row] * B[(k + lane) * N + col];

            // Warp-level reduction
            for (int offset = 16; offset > 0; offset /= 2) {
                partialSum += __shfl_down_sync(0xffffffff, partialSum, offset);
            }
            sum += partialSum;
        }

        if (lane == 0) {
            C[row * N + col] = sum;
        }
    }
}

// The forward function exposed via PyBind11.
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B.
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors and of type float32.
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N).
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    dim3 blockDim(32, 16);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x,
                 (N + blockDim.y - 1) / blockDim.y);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    warpOptimizedReductionKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B with warp-level reduction (CUDA)");
}
