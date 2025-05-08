#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel to compute C = A.T * B using a 1D grid-stride loop.
// A: Tensor of shape (K, M) stored as (K, M), where element A(k, i) is at A[k * M + i].
// B: Tensor of shape (K, N) stored as (K, N), where element B(k, j) is at B[k * N + j].
// C: Tensor of shape (M, N) stored in row-major order, where C(i, j) is at C[i * N + j].
// This design maps the output matrix to a 1D array and uses a uniform loop without conditional divergence
// in the inner computation, which minimizes warp divergence.
__global__ void linearNoDivKernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int K,
                                    int M,
                                    int N) {
    int total = M * N;
    // Use a grid-stride loop over all elements of C
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        // Calculate row (i) and column (j) for the output element
        int i = idx / N;
        int j = idx % N;
        float sum = 0.0f;
        // Uniform loop over K without any branch divergence
        for (int k = 0; k < K; ++k) {
            sum += A[k * M + i] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

// The forward function exposed via PyBind11.
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors and of type float32
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N)
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    int total = M * N;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the kernel
    linearNoDivKernel<<<blocks, threads>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B with minimal warp divergence (CUDA)");
}
