#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel using __ldg() for read-only global memory loads
// and loop unrolling to align accesses to 128-bit boundaries.
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            // Number of iterations from col to row (inclusive)
            int n_iters = row - col + 1;
            int k = col;
            // Unroll loop with factor of 4 for potential 128-bit aligned loads
            int unroll = n_iters / 4;
            for (int i = 0; i < unroll; i++) {
                float a0 = __ldg(&A[row * N + k]);
                float a1 = __ldg(&A[row * N + k + 1]);
                float a2 = __ldg(&A[row * N + k + 2]);
                float a3 = __ldg(&A[row * N + k + 3]);

                float b0 = __ldg(&B[k * N + col]);
                float b1 = __ldg(&B[(k + 1) * N + col]);
                float b2 = __ldg(&B[(k + 2) * N + col]);
                float b3 = __ldg(&B[(k + 3) * N + col]);

                sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
                k += 4;
            }
            // Process remaining elements
            for (; k <= row; k++) {
                sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
            }

            C[row * N + col] = sum;
        }
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Use 32x32 thread blocks to match warp size and encourage 128-bit aligned accesses
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + 31) / 32, (N + 31) / 32);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication with __ldg() optimizations (CUDA)");
}
