#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

__global__ void optimized_hybrid_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int row = by + ty;
    const int col = bx + tx;
    
    float sum = 0.0f;
    
    // Early exit condition for upper triangle
    if (row < col || row >= N || col >= N) {
        if (row < N && col < N) {
            C[row * N + col] = 0.0f;
        }
        return;
    }

    // Calculate the starting tile to avoid unnecessary computations
    const int start_tile = col / TILE_SIZE;
    const int end_tile = (row + TILE_SIZE - 1) / TILE_SIZE;
    
    #pragma unroll 4
    for (int tile = start_tile; tile <= end_tile; ++tile) {
        // Collaborative loading using vectorized loads when possible
        if ((tile * TILE_SIZE + tx) <= row) {
            As[ty][tx] = A[row * N + (tile * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((tile * TILE_SIZE + ty) < N) {
            Bs[ty][tx] = B[(tile * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum for this tile using loop unrolling
        if ((tile * TILE_SIZE) >= col) {
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; k += 4) {
                sum += As[ty][k] * Bs[k][tx];
                sum += As[ty][k+1] * Bs[k+1][tx];
                sum += As[ty][k+2] * Bs[k+2][tx];
                sum += As[ty][k+3] * Bs[k+3][tx];
            }
        } else {
            // Handle boundary case
            for (int k = 0; k < TILE_SIZE; ++k) {
                if ((tile * TILE_SIZE + k) >= col) {
                    sum += As[ty][k] * Bs[k][tx];
                }
            }
        }
        
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    optimized_hybrid_triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized hybrid triangular matrix multiplication (CUDA)");
}