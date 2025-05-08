#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// CUDA kernel for computing C = A.T * B using stride loops.
// A: shape (K, M), B: shape (K, N), C: shape (M, N)
// A is stored as (K, M) so element (k, i) is at A[k * M + i]
// B is stored as (K, N) so element (k, j) is at B[k * N + j]
// C is stored in row-major order: element (i, j) at C[i * N + j]
__global__ void strideOptimizedKernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int K,
                                        int M,
                                        int N) {
    // Calculate starting indices for this thread
    int i_start = blockIdx.x * blockDim.x + threadIdx.x;
    int j_start = blockIdx.y * blockDim.y + threadIdx.y;
    // Compute the strides along each dimension
    int stride_i = gridDim.x * blockDim.x;
    int stride_j = gridDim.y * blockDim.y;

    // Loop over the output matrix C with proper stride to cover all elements
    for (int i = i_start; i < M; i += stride_i) {
        for (int j = j_start; j < N; j += stride_j) {
            float sum = 0.0f;
            // Multiply across dimension K
            for (int k = 0; k < K; ++k) {
                sum += A[k * M + i] * B[k * N + j];
            }
            C[i * N + j] = sum;
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
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // Dimensions: A is (K, M) and B is (K, N)
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N)
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define thread block and grid sizes. Using a block of 16x16 threads.
    const int THREADS = 16;
    dim3 blockDim(THREADS, THREADS);
    dim3 gridDim((M + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS);

    // Get raw pointers to the data
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the CUDA kernel using stride loops to cover the entire output matrix
    strideOptimizedKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B with stride loops optimization (CUDA)");
}
