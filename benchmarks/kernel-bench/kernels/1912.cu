#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define BLOCK_ROWS 8

__global__ void aligned_triangular_mm_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int row = by + ty;
    const int col = bx + tx;
    
    float sum = 0.0f;
    
    // Only compute for lower triangular part
    if (row >= col && row < N && col < N) {
        // Process the matrix in aligned chunks where possible
        for (int t = col / TILE_SIZE; t <= row / TILE_SIZE; ++t) {
            const int tile_x = t * TILE_SIZE + tx;
            const int tile_y = t * TILE_SIZE + ty;
            
            // Load data into shared memory using __ldg for read-only access
            if (tile_x <= row && (by + ty) < N) {
                As[ty][tx] = __ldg(&A[row * N + tile_x]);
            } else {
                As[ty][tx] = 0.0f;
            }
            
            if (tile_y <= N && col < N) {
                Bs[ty][tx] = __ldg(&B[tile_y * N + col]);
            } else {
                Bs[ty][tx] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial results
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; ++k) {
                if ((t * TILE_SIZE + k) >= col && (t * TILE_SIZE + k) <= row) {
                    sum += As[ty][k] * Bs[k][tx];
                }
            }
            
            __syncthreads();
        }
        
        // Store the result
        if (row < N && col < N) {
            C[row * N + col] = sum;
        }
    } else if (row < N && col < N) {
        // Set upper triangular part to zero
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Configure kernel dimensions for better occupancy
    dim3 threadsPerBlock(TILE_SIZE / 2, BLOCK_ROWS);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    aligned_triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Aligned triangular matrix multiplication (CUDA)");
}