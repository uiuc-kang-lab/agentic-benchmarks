#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the warp size for our kernel
#define WARP_SIZE 32

// This kernel computes C = tril(A * B) for lower-triangular matrices A and B.
// It uses block-level culling and warp-level checks for early exit in regions that
// are entirely in the upper triangle to reduce divergence.
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    // Compute global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check boundary conditions
    if (row >= N || col >= N) return;

    // Block-level culling: Determine block boundaries
    int block_row_start = blockIdx.y * blockDim.y;
    int block_col_start = blockIdx.x * blockDim.x;
    int block_row_end = block_row_start + blockDim.y - 1;
    int block_col_end = block_col_start + blockDim.x - 1;

    // If the entire block is in the upper triangular region, skip compute
    // All threads in this block satisfy: row < col
    if (block_row_end < block_col_start) {
        C[row * N + col] = 0.f;
        return;
    }

    // If the entire block is in the lower triangular region, compute without extra checks
    // All threads satisfy: row >= col
    if (block_row_start >= block_col_end) {
        float sum = 0.f;
        #pragma unroll 4
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
        return;
    }

    // For blocks that straddle the diagonal, we use a warp-level check to reduce divergence
    int warp_row = row & ~(WARP_SIZE - 1);
    int warp_col = col & ~(WARP_SIZE - 1);
    if (warp_row < warp_col) {
        C[row * N + col] = 0.f;
        return;
    }

    // Finally, do an individual thread check for diagonal blocks
    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    float sum = 0.f;
    // Compute the multiplication only over the applicable range
    for (int k = col; k <= row; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// The PyTorch interface: verifies input properties and launches the kernel
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

    // Use 32x32 thread blocks to align with warp boundaries for optimal execution
    dim3 threadsPerBlock(WARP_SIZE, WARP_SIZE);
    dim3 numBlocks((N + WARP_SIZE - 1) / WARP_SIZE, (N + WARP_SIZE - 1) / WARP_SIZE);

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
    m.def("forward", &forward, "Triangular Matrix Multiplication with warp and block-level optimizations (CUDA)");
}
