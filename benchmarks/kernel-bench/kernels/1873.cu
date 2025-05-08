#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Calculate global thread indices
    const int global_x = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    const int global_y = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    
    // Each thread computes one element of C
    if (global_y < N && global_x < N) {
        const int row = global_y;
        const int col = global_x;
        
        // Only compute for lower triangular part
        if (row >= col) {
            float sum = 0.0f;
            
            // Calculate start and end points for dot product
            const int start_k = col;
            const int end_k = row;
            
            // Unrolled loop for better instruction-level parallelism
            int k = start_k;
            #pragma unroll 4
            for (; k <= end_k - 4; k += 4) {
                sum += A[row * N + k] * B[k * N + col];
                sum += A[row * N + (k+1)] * B[(k+1) * N + col];
                sum += A[row * N + (k+2)] * B[(k+2) * N + col];
                sum += A[row * N + (k+3)] * B[(k+3) * N + col];
            }
            
            // Handle remaining elements
            for (; k <= end_k; k++) {
                sum += A[row * N + k] * B[k * N + col];
            }
            
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
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

    // Calculate grid dimensions
    dim3 threadsPerBlock(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 numBlocks(
        (N + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
        (N + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y
    );

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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}