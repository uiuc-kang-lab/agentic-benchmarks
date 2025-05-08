#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// This kernel computes C = A.T * B using a grid-stride loop to cover the entire output
// matrix even if there are fewer threads than elements. The inner loop over K is unrolled
// by a factor of 4 to reduce loop overhead. Boundary conditions are checked properly to
// ensure correctness.
__global__ void linearStrideUnrollKernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int K, int M, int N) {
    int total = M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < total; idx += stride) {
        int i = idx / N;    // row index for C (and column index for A)
        int j = idx % N;    // column index for C (and B)
        float sum = 0.0f;
        int k = 0;
        // Loop unrolling by a factor of 4 for efficiency
        for (; k <= K - 4; k += 4) {
            sum += A[(k + 0) * M + i] * B[(k + 0) * N + j]
                + A[(k + 1) * M + i] * B[(k + 1) * N + j]
                + A[(k + 2) * M + i] * B[(k + 2) * N + j]
                + A[(k + 3) * M + i] * B[(k + 3) * N + j];
        }
        // Handle remaining elements if K is not a multiple of 4
        for (; k < K; ++k) {
            sum += A[k * M + i] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

// Forward function exposed via PyBind11
// A: Tensor of shape (K, M) [CUDA, float32]
// B: Tensor of shape (K, N) [CUDA, float32]
// Returns: C, Tensor of shape (M, N) computed as C = A.T * B

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    int threads = 256;
    int blocks = (M * N + threads - 1) / threads;

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    linearStrideUnrollKernel<<<blocks, threads>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using grid-stride loop with unrolling (CUDA)");
}
