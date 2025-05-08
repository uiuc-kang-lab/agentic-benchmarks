#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses 2D grid-stride loops to cover the entire matrix.
// Each thread computes one or more elements of the output matrix C corresponding to the product of lower triangular matrices A and B.
// For a given output element C[i,j] with i >= j, the kernel computes the sum for k from j to i, and sets C[i,j] = sum.
// If i < j, the element is set to 0. Stride loops ensure correct boundary handling for matrices larger than the grid.

__global__ void strided_triangular_mm(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N) {
    // Determine initial row and col indices for this thread
    int row_start = blockIdx.y * blockDim.y + threadIdx.y;
    int col_start = blockIdx.x * blockDim.x + threadIdx.x;

    // Compute stride for rows and columns
    int stride_row = gridDim.y * blockDim.y;
    int stride_col = gridDim.x * blockDim.x;

    // Loop over rows assigned to this thread using grid-stride loop
    for (int i = row_start; i < N; i += stride_row) {
        // Loop over columns assigned to this thread using grid-stride loop
        for (int j = col_start; j < N; j += stride_col) {
            // Only compute the lower triangular part; upper triangular elements are zero
            if (i < j) {
                C[i * N + j] = 0.0f;
            } else {
                float sum = 0.0f;
                // Compute the dot-product for the lower triangular multiplication
                // Only indices from j to i contribute
                #pragma unroll
                for (int k = j; k <= i; ++k) {
                    sum += A[i * N + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
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

    // Define a 2D block and grid configuration
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    strided_triangular_mm<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Strided triangular matrix multiplication (CUDA)");
}
