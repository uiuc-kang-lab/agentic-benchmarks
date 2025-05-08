#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib> // for abs

// This kernel computes C = tril(A * B) for lower triangular matrices A and B.
// It minimizes warp divergence by categorizing each block into one of three cases:
// 1. Entire block is in the upper triangular region -> write 0 without further checks.
// 2. Entire block is in the lower triangular region -> all threads compute the full sum without extra branching.
// 3. Diagonal block (straddling the matrix diagonal) -> use branchless arithmetic to compute the number of loop iterations,
//    eliminating conditional branches within the inner loop and ensuring uniform control flow across the warp.

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    // Compute global row and column indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;

    // Determine block boundaries
    int block_row_start = blockIdx.y * blockDim.y;
    int block_col_start = blockIdx.x * blockDim.x;
    int block_row_end = block_row_start + blockDim.y - 1;
    int block_col_end = block_col_start + blockDim.x - 1;

    // Case 1: Entire block is in the upper triangular region.
    // In this case, for every thread, the entire block satisfies row < col, so set C to 0.
    if (block_row_end < block_col_start) {
        C[row * N + col] = 0.0f;
        return;
    }

    // Case 2: Entire block is in the lower triangular region.
    // Here, all threads satisfy row >= col so we can perform the summation without additional conditional checks.
    if (block_row_start >= block_col_end) {
        float sum = 0.0f;
        #pragma unroll
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
        return;
    }

    // Case 3: Diagonal block (straddling the matrix diagonal).
    // Here, threads in the same block may be both valid (row >= col) and invalid (row < col).
    // To avoid divergent branching within warps, we compute the number of iterations in a branchless manner:
    //   num_iter = max(row - col + 1, 0) = (diff + abs(diff)) >> 1, where diff = row - col + 1
    int diff = row - col + 1;
    int num_iter = (diff + abs(diff)) >> 1;  // branchless max(diff, 0)
    float sum = 0.0f;
    #pragma unroll
    for (int t = 0; t < num_iter; t++) {
        int k = col + t;
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// PyTorch interface
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Launch with 32x32 thread blocks to align with warp size
    const int block_dim = 32;
    dim3 threads(block_dim, block_dim);
    dim3 blocks((N + block_dim - 1) / block_dim, (N + block_dim - 1) / block_dim);

    triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Branchless Triangular Matrix Multiplication (CUDA)");
}
