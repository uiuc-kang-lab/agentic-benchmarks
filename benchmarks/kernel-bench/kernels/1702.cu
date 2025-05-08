#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 32
#define BLOCK_SIZE WARP_SIZE

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= N || col >= N) return;
    
    const int block_row_start = blockIdx.y * blockDim.y;
    const int block_col_start = blockIdx.x * blockDim.x;
    const int block_row_end = block_row_start + blockDim.y - 1;
    const int block_col_end = block_col_start + blockDim.x - 1;

    if (block_row_end < block_col_start) {
        C[row * N + col] = 0.f;
        return;
    }

    if (block_row_start >= block_col_end) {
        float sum = 0.f;
        
        for (int tile = 0; tile < (row - col + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
            const int tile_start = col + tile * TILE_SIZE;
            
            if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
                const int load_row = block_row_start + threadIdx.y;
                const int load_col = tile_start + threadIdx.x;
                if (load_row < N && load_col < N) {
                    As[threadIdx.y][threadIdx.x] = __ldg(&A[load_row * N + load_col]);
                    Bs[threadIdx.y][threadIdx.x] = __ldg(&B[load_col * N + block_col_start + threadIdx.x]);
                }
            }
            __syncthreads();

            #pragma unroll 8
            for (int k = 0; k < TILE_SIZE; ++k) {
                if (tile_start + k <= row)
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
            __syncthreads();
        }
        C[row * N + col] = sum;
        return;
    }

    const int warp_row = row & ~(WARP_SIZE - 1);
    const int warp_col = col & ~(WARP_SIZE - 1);
    if (warp_row < warp_col || row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    float sum = 0.f;
    for (int tile = 0; tile < (row - col + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        const int tile_start = col + tile * TILE_SIZE;
        
        if (threadIdx.x < TILE_SIZE && threadIdx.y < TILE_SIZE) {
            const int load_row = block_row_start + threadIdx.y;
            const int load_col = tile_start + threadIdx.x;
            if (load_row < N && load_col < N) {
                As[threadIdx.y][threadIdx.x] = __ldg(&A[load_row * N + load_col]);
                Bs[threadIdx.y][threadIdx.x] = __ldg(&B[load_col * N + block_col_start + threadIdx.x]);
            }
        }
        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; ++k) {
            if (tile_start + k <= row)
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Inputs must have same dimensions");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}