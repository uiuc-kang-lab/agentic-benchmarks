#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use 32 threads per warp
#define WARP_SIZE 32

// This kernel combines grid-striding loops from Kernel 2 with the read-only __ldg loads
// and loop unrolling from Kernel 1. It processes multiple elements per thread so that
// the entire matrix is covered even when grid dimensions are smaller than the problem size.
__global__ void optimized_triangular_mm_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                int N) {
    // Calculate starting indices based on block and thread indices
    int row_start = blockIdx.y * blockDim.y + threadIdx.y;
    int col_start = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute strides across the matrix for rows and columns
    int stride_row = blockDim.y * gridDim.y;
    int stride_col = blockDim.x * gridDim.x;

    // Each thread processes several output elements in a grid-strided loop
    for (int row = row_start; row < N; row += stride_row) {
        for (int col = col_start; col < N; col += stride_col) {
            // For a triangular matrix multiplication, if row < col then output is 0
            if (row < col) {
                C[row * N + col] = 0.f;
            } else {
                float sum = 0.f;
                // Unroll the inner loop and use __ldg to leverage read-only cache
                #pragma unroll 4
                for (int k = col; k <= row; ++k) {
                    sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
                }
                C[row * N + col] = sum;
            }
        }
    }
}

// Host function: performs checks on tensors and launches the optimized kernel
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

    // Use 32x32 threads per block
    dim3 threadsPerBlock(WARP_SIZE, WARP_SIZE);
    dim3 numBlocks((N + WARP_SIZE - 1) / WARP_SIZE, (N + WARP_SIZE - 1) / WARP_SIZE);

    optimized_triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Optimized Triangular Matrix Multiplication with __ldg and grid-striding (CUDA)");
}
