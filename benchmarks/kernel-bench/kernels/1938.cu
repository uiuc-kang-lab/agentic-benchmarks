#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define VECTOR_SIZE 4  // For float4 vectorized loads

__global__ void triangular_mm_kernel_vectorized(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Early exit for upper triangular part
    if (row < N && col < N && row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    
    // Calculate aligned positions for vectorized loads
    const int aligned_col = col & ~(VECTOR_SIZE - 1);
    
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        const int tile_start = tile * TILE_SIZE;
        
        // Load tile of matrix A using vectorized loads where possible
        if (row < N && threadIdx.x < TILE_SIZE) {
            const int a_col = tile_start + threadIdx.x;
            if (a_col < N) {
                if ((row * N + a_col) % VECTOR_SIZE == 0 && a_col + VECTOR_SIZE <= N) {
                    // Vectorized load for aligned addresses
                    float4 a_vec = *reinterpret_cast<const float4*>(&A[row * N + a_col]);
                    As[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + a_col]);
                } else {
                    // Regular load for unaligned addresses
                    As[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + a_col]);
                }
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
        
        // Load tile of matrix B using vectorized loads where possible
        if (col < N && threadIdx.y < TILE_SIZE) {
            const int b_row = tile_start + threadIdx.y;
            if (b_row < N) {
                if ((b_row * N + aligned_col) % VECTOR_SIZE == 0 && col + VECTOR_SIZE <= N) {
                    // Vectorized load for aligned addresses
                    float4 b_vec = *reinterpret_cast<const float4*>(&B[b_row * N + aligned_col]);
                    Bs[threadIdx.y][threadIdx.x] = __ldg(&B[b_row * N + col]);
                } else {
                    // Regular load for unaligned addresses
                    Bs[threadIdx.y][threadIdx.x] = __ldg(&B[b_row * N + col]);
                }
            } else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
        }
        
        __syncthreads();
        
        if (row < N && col < N && row >= col) {
            // Compute partial dot product for this tile
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                const int global_k = tile_start + k;
                if (global_k >= col && global_k <= row && global_k < N) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
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

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    // Launch kernel with optimal block configuration
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel_vectorized<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized lower triangular matrix multiplication (CUDA)");
}