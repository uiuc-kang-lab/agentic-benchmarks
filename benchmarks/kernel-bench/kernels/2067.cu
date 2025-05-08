#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define BLOCK_SIZE 256
#define ELEMENTS_PER_THREAD 8

__global__ void hybrid_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N) {
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int tile_row = blockIdx.y;
    const int tile_col = blockIdx.x;
    const int row = tile_row * TILE_SIZE + threadIdx.y;
    const int col = tile_col * TILE_SIZE + threadIdx.x;

    // Early exit for upper triangular part
    if (tile_row * TILE_SIZE < tile_col * TILE_SIZE) {
        if (row < N && col < N) {
            C[row * N + col] = 0.0f;
        }
        return;
    }

    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = tile_col; t <= tile_row; t++) {
        // Collaborative loading of tiles into shared memory
        if (row < N && (t * TILE_SIZE + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((t * TILE_SIZE + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial product for this tile
        if (row < N && col < N && row >= col) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < N && col < N && row >= col) {
        C[row * N + col] = sum;
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    hybrid_triangular_mm_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}