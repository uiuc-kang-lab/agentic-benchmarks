#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel to compute C = tril(A * B) for lower triangular matrices A and B using warp-level primitives.
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    // Compute the row and column indices for this thread.
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        // Only compute values for the lower triangular part.
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            // For lower triangular matrices, A[i,k] is nonzero only if k <= i
            // and B[k,j] is nonzero only if j <= k.
            // Hence, only indices k from col to row contribute.
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B[k * N + col];
            }

            // Use warp-level reduction to sum the values within a warp
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            }

            // Write the result from the first thread of each warp
            if (threadIdx.x % warpSize == 0) {
                C[row * N + col] = sum;
            }
        }
    }
}

// C++ interface exposed to PyTorch.
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

    // Define block and grid dimensions.
    const int threads = 32;  // Using a higher thread count per block for better warp utilization.
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    // Launch the CUDA kernel.
    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    // Check for kernel launch errors.
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}
