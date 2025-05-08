#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N) return;
    
    if (row >= col) {
        float sum = 0.0f;
        
        // Process elements in tiles of 4
        const int TILE_SIZE = 4;
        int k = col;
        
        // Main loop with manual unrolling for tiles of 4
        #pragma unroll 1
        for (; k <= row - (TILE_SIZE - 1); k += TILE_SIZE) {
            // Manual unroll of 4 iterations
            sum += A[row * N + k] * B[k * N + col];
            sum += A[row * N + (k+1)] * B[(k+1) * N + col];
            sum += A[row * N + (k+2)] * B[(k+2) * N + col];
            sum += A[row * N + (k+3)] * B[(k+3) * N + col];
        }
        
        // Handle remaining elements
        #pragma unroll
        for (; k <= row; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        
        C[row * N + col] = sum;
    } else {
        C[row * N + col] = 0.0f;
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

    // Use 32x32 thread blocks to maximize occupancy
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

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