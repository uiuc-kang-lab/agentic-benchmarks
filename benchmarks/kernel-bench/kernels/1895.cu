#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by + ty;
    int col = bx + tx;
    
    float sum = 0.0f;
    
    // Only compute for lower triangular portion
    if (row >= col && row < N && col < N) {
        // Loop over tiles
        for (int t = col; t <= row; t += TILE_SIZE) {
            // Load tile from A - only load if within bounds and lower triangular
            if (t + tx <= row && row < N) {
                As[ty][tx] = A[row * N + (t + tx)];
            } else {
                As[ty][tx] = 0.0f;
            }
            
            // Load tile from B - only load if within bounds and lower triangular
            if (col < N && t + ty >= col) {
                Bs[ty][tx] = B[(t + ty) * N + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            
            // Ensure all threads have loaded their data
            __syncthreads();
            
            // Compute partial dot product for this tile
            for (int k = 0; k < TILE_SIZE; k++) {
                if (t + k <= row) {
                    sum += As[ty][k] * Bs[k][tx];
                }
            }
            
            // Ensure computation is complete before loading next tile
            __syncthreads();
        }
        
        // Write result
        C[row * N + col] = sum;
    } else if (row < col && row < N && col < N) {
        // Upper triangular portion is zero
        C[row * N + col] = 0.0f;
    }
}

// C++ interface exposed to PyTorch
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                   (N + TILE_SIZE - 1) / TILE_SIZE);

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