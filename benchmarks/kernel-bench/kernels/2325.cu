#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 32;
const int VECTOR_SIZE = 4;  // Process 4 elements at once

__global__ void matmul_optimized_kernel(const float* __restrict__ A, 
                                      const float* __restrict__ B, 
                                      float* __restrict__ C, 
                                      int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    // Use float4 for coalesced memory access
    float4 a_reg, b_reg;
    float c_val = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;

        // Vectorized loading of A using __ldg
        if (row < M && (k_offset + threadIdx.x) < K && threadIdx.x % VECTOR_SIZE == 0) {
            a_reg = *reinterpret_cast<const float4*>(__ldg(&A[row * K + k_offset + threadIdx.x]));
            As[threadIdx.y][threadIdx.x] = a_reg.x;
            if (threadIdx.x + 1 < TILE_SIZE) As[threadIdx.y][threadIdx.x + 1] = a_reg.y;
            if (threadIdx.x + 2 < TILE_SIZE) As[threadIdx.y][threadIdx.x + 2] = a_reg.z;
            if (threadIdx.x + 3 < TILE_SIZE) As[threadIdx.y][threadIdx.x + 3] = a_reg.w;
        } else if (row < M && (k_offset + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + k_offset + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Vectorized loading of B using __ldg
        if (col < N && (k_offset + threadIdx.y) < K && threadIdx.y % VECTOR_SIZE == 0) {
            b_reg = *reinterpret_cast<const float4*>(__ldg(&B[col * K + k_offset + threadIdx.y]));
            Bs[threadIdx.y][threadIdx.x] = b_reg.x;
            if (threadIdx.y + 1 < TILE_SIZE) Bs[threadIdx.y + 1][threadIdx.x] = b_reg.y;
            if (threadIdx.y + 2 < TILE_SIZE) Bs[threadIdx.y + 2][threadIdx.x] = b_reg.z;
            if (threadIdx.y + 3 < TILE_SIZE) Bs[threadIdx.y + 3][threadIdx.x] = b_reg.w;
        } else if (col < N && (k_offset + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[col * K + k_offset + threadIdx.y]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 4) {
            c_val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            c_val += As[threadIdx.y][k+1] * Bs[k+1][threadIdx.x];
            c_val += As[threadIdx.y][k+2] * Bs[k+2][threadIdx.x];
            c_val += As[threadIdx.y][k+3] * Bs[k+3][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = c_val;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());
    
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);
    
    matmul_optimized_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    return C;
}