#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];
    
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int row = by + ty;
    const int col = bx + tx;
    
    float sum = 0.0f;
    
    // Only compute for lower triangular portion
    if (row >= col && row < N && col < N) {
        // Preload first tile
        if (row < N && tx < N)
            As[0][ty][tx] = A[row * N + tx];
        else
            As[0][ty][tx] = 0.0f;
            
        if (ty < N && col < N)
            Bs[0][ty][tx] = B[ty * N + col];
        else
            Bs[0][ty][tx] = 0.0f;
            
        __syncthreads();
        
        // Loop over tiles
        for (int t = 0; t < N; t += TILE_SIZE) {
            int next_tile = (t + TILE_SIZE < N) ? 1 : 0;
            int current = 0;
            
            // Preload next tile if it exists
            if (t + TILE_SIZE < N) {
                if (row < N && (t + TILE_SIZE + tx) < N)
                    As[next_tile][ty][tx] = A[row * N + (t + TILE_SIZE + tx)];
                else
                    As[next_tile][ty][tx] = 0.0f;
                    
                if ((t + TILE_SIZE + ty) < N && col < N)
                    Bs[next_tile][ty][tx] = B[(t + TILE_SIZE + ty) * N + col];
                else
                    Bs[next_tile][ty][tx] = 0.0f;
            }
            
            // Compute using current tile while next tile is loading
            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; k++) {
                int global_k = t + k;
                if (global_k >= col && global_k <= row && global_k < N) {
                    sum += As[current][ty][k] * Bs[current][k][tx];
                }
            }
            
            __syncthreads();
            current = next_tile;
        }
        
        // Write result
        C[row * N + col] = sum;
    } else if (row < col && row < N && col < N) {
        // Set upper triangular portion to zero
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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, 
              (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<grid, threads>>>(
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