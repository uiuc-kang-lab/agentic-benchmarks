#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define TILE_SIZE 32
#define WARP_SIZE 32
#define BLOCK_SIZE 256

__global__ void hybrid_triangular_mm_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          const int N) {
    // For larger matrices (N > TILE_SIZE * 2), use tiled approach
    if (N > TILE_SIZE * 2) {
        int row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int col = blockIdx.x * TILE_SIZE + threadIdx.x;

        if (row < N && col < N) {
            if (row < col) {
                C[row * N + col] = 0.f;
            } else {
                float sum = 0.f;
                // Use shared memory for tiles
                __shared__ float As[TILE_SIZE][TILE_SIZE];
                __shared__ float Bs[TILE_SIZE][TILE_SIZE];
                
                for (int tile = col / TILE_SIZE; tile <= row / TILE_SIZE; ++tile) {
                    // Collaborative loading of tiles
                    int k = tile * TILE_SIZE + threadIdx.x;
                    if (k <= row) {
                        As[threadIdx.y][threadIdx.x] = A[row * N + k];
                        Bs[threadIdx.y][threadIdx.x] = B[k * N + col];
                    } else {
                        As[threadIdx.y][threadIdx.x] = 0.0f;
                        Bs[threadIdx.y][threadIdx.x] = 0.0f;
                    }
                    __syncthreads();

                    // Compute partial sum for this tile
                    #pragma unroll
                    for (int k = 0; k < TILE_SIZE; ++k) {
                        if (tile * TILE_SIZE + k <= row && tile * TILE_SIZE + k >= col) {
                            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                        }
                    }
                    __syncthreads();
                }
                C[row * N + col] = sum;
            }
        }
    } 
    // For smaller matrices, use warp-uniform approach
    else {
        unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int total_elements = N * (N + 1) / 2;
        
        if (idx >= total_elements) return;

        int row = (int)((-1 + sqrt(8.0f * idx + 1)) / 2);
        int col = idx - row * (row + 1) / 2;
        
        float sum = 0.0f;
        #pragma unroll 4
        for (int k = col; k <= row; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrix dimension mismatch");
    
    const int N = A.size(0);
    auto C = torch::zeros_like(A);

    if (N > TILE_SIZE * 2) {
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
        hybrid_triangular_mm_kernel<<<blocks, threads>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    } else {
        const int total_elements = N * (N + 1) / 2;
        const int grid_size = (total_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        hybrid_triangular_mm_kernel<<<grid_size, BLOCK_SIZE>>>(
            A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid triangular matrix multiplication (CUDA)");
}