#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel that leverages optimal block sizes and memory coalescing for triangular matrix multiplication.
// It maps threadIdx.x to the column dimension for coalesced B accesses and uses an optimal block configuration
// that balances warp occupancy and memory throughput for A and B accesses.

// Kernel computes C = triangular multiplication of A and B where only the lower-triangular portion is computed.
// For each element (row, col), if row < col, C is set to 0; otherwise, it computes:
//   C[row, col] = sum_{k=col}^{row} A[row, k] * B[k, col]

__global__ void optimized_coalesced_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    // Map thread.x to column and thread.y to row
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= N || col >= N) return;

    // If above the diagonal, assign zero
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    // Compute the dot-product for the triangular portion
    float sum = 0.0f;
    for (int k = col; k <= row; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}

// PyTorch forward function wrapper
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Chosen block size: using 32 threads in x for warp alignment (coalesced accesses for B) and 16 in y for balanced occupancy
    const int bx = 32, by = 16;
    dim3 threads(bx, by);
    dim3 blocks((N + bx - 1) / bx, (N + by - 1) / by);

    optimized_coalesced_triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Optimized coalesced triangular matmul (CUDA)");
}
