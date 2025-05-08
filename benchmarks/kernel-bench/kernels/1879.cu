#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Calculate total number of elements in lower triangle
    const int total_elements = (N * (N + 1)) / 2;
    
    // Get global thread ID
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < total_elements) {
        // Convert linear index to row,col using quadratic formula
        // solving k = (row * (row + 1))/2 + col
        // where k is the linear index in the lower triangle
        int row = (-1 + sqrt(1 + 8 * (float)tid)) / 2;
        int col = tid - (row * (row + 1)) / 2;
        
        // Compute matrix multiplication for this element
        float sum = 0.0f;
        
        #pragma unroll 8
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        // Store result
        C[row * N + col] = sum;
        
        // If we're on the diagonal, zero out the upper triangle elements in this column
        if (col == row) {
            #pragma unroll 4
            for (int i = 0; i < row; ++i) {
                C[i * N + row] = 0.0f;
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Calculate total number of elements in lower triangle
    const int total_elements = (N * (N + 1)) / 2;
    
    // Calculate grid dimensions
    const int numBlocks = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    triangular_mm_kernel<<<numBlocks, BLOCK_SIZE>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}