#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel leverages warp-level primitives for reduction, eliminating shared memory usage for the inner dot product.
// Each warp computes one output element in the lower triangular matrix multiplication:
// For a given element (row, col) with row >= col, we compute C[row, col] = sum_{k=col}^{row} A[row, k] * B[k, col].
// Each warp of 32 threads cooperatively processes the dot product in a strided manner, then uses __shfl_down_sync for the reduction.

__global__ void warp_reduce_triangular_mm_kernel(const float* __restrict__ A,
                                                  const float* __restrict__ B,
                                                  float* __restrict__ C,
                                                  int N) {
    // Each block is configured such that blockDim.x == 32 (one warp per computed element) and blockDim.y = number of warps per block.
    int lane = threadIdx.x;          // Lane within the warp (0-31)
    int warp_id = threadIdx.y;         // Warp id within this block

    // Map each warp to one output element of C
    // Global row index is computed from blockIdx.y and warp_id
    int row = blockIdx.y * blockDim.y + warp_id;
    // Global column index is directly given by blockIdx.x
    int col = blockIdx.x;

    if (row >= N || col >= N) return;

    // Only compute for lower triangular part; else, output 0
    if (row < col) {
        if (lane == 0)
            C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    // Each warp cooperatively computes the dot product over k = col to row.
    // Stride by warpSize (32) so that each thread handles a subset of indices.
    for (int k = col + lane; k <= row; k += 32) {
        sum += A[row * N + k] * B[k * N + col];
    }

    // Warp-level reduction using __shfl_down_sync
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Lane 0 writes the final sum
    if (lane == 0) {
        C[row * N + col] = sum;
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

    // Launch configuration:
    // - grid.x covers the column indices: one block per column (grid.x = N).
    // - grid.y covers the rows in groups of 'warp_count'.
    // Each block has blockDim.x = 32 (the warp width) and blockDim.y = warp_count (number of warps per block).
    int warp_count = 8;  // Tune this value as needed
    dim3 block(32, warp_count);
    dim3 grid(N, (N + warp_count - 1) / warp_count);

    warp_reduce_triangular_mm_kernel<<<grid, block>>>(
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
    m.def("forward", &forward, "Warp-level reduced lower triangular matrix multiplication (CUDA)");
}
