#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    // Calculate global indices
    const int gRow = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int gCol = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    // Local indices
    const int ly = threadIdx.y;
    const int lx = threadIdx.x;
    
    // Early exit if we're outside the matrix bounds
    if (gRow >= N || gCol >= N) return;
    
    // Handle upper triangular part directly
    if (gRow < gCol) {
        C[gRow * N + gCol] = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    
    // Calculate number of tiles needed
    const int numTiles = (min(gRow, N - 1) - gCol + BLOCK_SIZE) / BLOCK_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        const int tileStart = gCol + t * BLOCK_SIZE;
        
        // Load tile from A into shared memory
        if (gRow < N && (tileStart + lx) < N) {
            As[ly][lx] = A[gRow * N + (tileStart + lx)];
        } else {
            As[ly][lx] = 0.0f;
        }
        
        // Load tile from B into shared memory
        if ((tileStart + ly) < N && gCol < N) {
            Bs[ly][lx] = B[(tileStart + ly) * N + gCol];
        } else {
            Bs[ly][lx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot product
        #pragma unroll 8
        for (int k = 0; k < BLOCK_SIZE; k++) {
            if ((tileStart + k) >= gCol && (tileStart + k) <= gRow) {
                sum += As[ly][k] * Bs[k][lx];
            }
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    if (gRow >= gCol && gRow < N && gCol < N) {
        C[gRow * N + gCol] = sum;
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
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}