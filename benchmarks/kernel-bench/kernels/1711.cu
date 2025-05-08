#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 32
#define BLOCK_SIZE WARP_SIZE
#define VECTOR_SIZE 4  // float4 loads 4 elements at once

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+VECTOR_SIZE];  // Pad for alignment
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+VECTOR_SIZE];  // Pad for alignment
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N) return;

    // Early exit conditions for upper triangular part
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    
    // Aligned base pointers for vector loads
    const float4* A_aligned = reinterpret_cast<const float4*>(A + row * N);
    const float4* B_aligned = reinterpret_cast<const float4*>(B + col);
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        const int tile_start = t * TILE_SIZE;
        const int tile_end = min(tile_start + TILE_SIZE, N);
        
        if (tile_start > row) break;

        // Vector load for tile A
        if (threadIdx.x < TILE_SIZE/VECTOR_SIZE) {
            const int shared_col = threadIdx.x * VECTOR_SIZE;
            const int global_col = tile_start + shared_col;
            if (global_col < N) {
                float4 a4 = __ldg(&A_aligned[global_col/VECTOR_SIZE]);
                As[threadIdx.y][shared_col] = a4.x;
                As[threadIdx.y][shared_col+1] = a4.y;
                As[threadIdx.y][shared_col+2] = a4.z;
                As[threadIdx.y][shared_col+3] = a4.w;
            }
        }

        // Vector load for tile B
        if (threadIdx.y < TILE_SIZE/VECTOR_SIZE) {
            const int shared_row = threadIdx.y * VECTOR_SIZE;
            const int global_row = tile_start + shared_row;
            if (global_row < N) {
                float4 b4 = __ldg(&B_aligned[global_row * (N/VECTOR_SIZE)]);
                Bs[shared_row][threadIdx.x] = b4.x;
                Bs[shared_row+1][threadIdx.x] = b4.y;
                Bs[shared_row+2][threadIdx.x] = b4.z;
                Bs[shared_row+3][threadIdx.x] = b4.w;
            }
        }
        
        __syncthreads();
        
        if (row >= col) {
            const int k_start = max(tile_start, col);
            const int k_end = min(tile_end, row + 1);
            
            #pragma unroll 8
            for (int k = k_start; k < k_end; k++) {
                const int k_local = k - tile_start;
                sum += As[threadIdx.y][k_local] * Bs[k_local][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    // Aligned store of result
    if (row >= col) {
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Input dimensions must match");
    TORCH_CHECK(A.size(0) % VECTOR_SIZE == 0, "Matrix dimension must be divisible by 4 for aligned access");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Vectorized Aligned Triangular Matrix Multiplication (CUDA)");
}