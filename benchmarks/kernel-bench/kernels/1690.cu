#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define warp size
#define WARP_SIZE 32

// This kernel computes C = tril(A * B) for lower triangular matrices A and B.
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    // Check if entire block is in the lower or upper triangle
    if (threadIdx.y < blockDim.y && threadIdx.x < blockDim.x) {
        for (int b_row = blockIdx.y * blockDim.y; b_row < min(N, (blockIdx.y + 1) * blockDim.y); ++b_row) {
            for (int b_col = blockIdx.x * blockDim.x; b_col < min(N, (blockIdx.x + 1) * blockDim.x); ++b_col) {
                if (b_row < b_col) {
                    C[b_row * N + b_col] = 0.f;
                }
            }
        }
    }

    if (row >= col) {
        float sum = 0.f;
        #pragma unroll 4
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    } else {
        C[row * N + col] = 0.f;
    }
}

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

    // Test different block sizes to find optimal setting
    int blockSize = 256;  // Initial choice, can be adapted as needed
    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 numBlocks((N + blockSize - 1) / blockSize, (N + blockSize - 1) / blockSize);

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
    m.def("forward", &forward, "Adaptive Block Size Triangular Matrix Multiplication (CUDA)");
}
