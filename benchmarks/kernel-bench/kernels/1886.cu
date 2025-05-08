#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 64
#define INNER_TILE 4
#define WARP_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Register array for accumulation
    float acc[INNER_TILE][INNER_TILE] = {{0.0f}};
    
    // Calculate number of tiles needed
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    #pragma unroll 4
    for (int tile = 0; tile <= blockIdx.y; ++tile) {
        const int tile_start = tile * TILE_SIZE;
        
        // Load tiles into shared memory with unrolled accesses
        #pragma unroll 2
        for (int i = 0; i < TILE_SIZE; i += WARP_SIZE) {
            if ((row + i) < N && (tile_start + threadIdx.x) < N) {
                s_A[threadIdx.y + i][threadIdx.x] = A[(row + i) * N + tile_start + threadIdx.x];
            } else {
                s_A[threadIdx.y + i][threadIdx.x] = 0.0f;
            }
            
            if ((tile_start + threadIdx.y + i) < N && col < N) {
                s_B[threadIdx.y + i][threadIdx.x] = B[(tile_start + threadIdx.y + i) * N + col];
            } else {
                s_B[threadIdx.y + i][threadIdx.x] = 0.0f;
            }
        }
        
        __syncthreads();
        
        if (row < N && col < N && row >= col) {
            const int k_start = max(tile_start, col);
            const int k_end = min(tile_start + TILE_SIZE, row + 1);
            
            // Fully unrolled inner computation loops
            #pragma unroll
            for (int k = k_start; k < k_end; k += INNER_TILE) {
                #pragma unroll
                for (int ki = 0; ki < INNER_TILE && (k + ki) < k_end; ++ki) {
                    #pragma unroll
                    for (int kj = 0; kj < INNER_TILE && (k + kj) < k_end; ++kj) {
                        acc[ki][kj] += s_A[threadIdx.y][k - tile_start + ki] * 
                                     s_B[k - tile_start + kj][threadIdx.x];
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Reduction of accumulated values
    if (row < N && col < N) {
        if (row >= col) {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < INNER_TILE; ++i) {
                #pragma unroll
                for (int j = 0; j < INNER_TILE; ++j) {
                    sum += acc[i][j];
                }
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

    dim3 threadsPerBlock(WARP_SIZE, WARP_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}