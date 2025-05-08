#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

__device__ __forceinline__ void load_tile_to_shared(float* shared_tile, 
                                                   const float* global_mat,
                                                   const int row, const int col, 
                                                   const int N, 
                                                   const int tile_row, const int tile_col) {
    if (row < N && col < N) {
        shared_tile[tile_row * TILE_SIZE + tile_col] = global_mat[row * N + col];
    } else {
        shared_tile[tile_row * TILE_SIZE + tile_col] = 0.0f;
    }
}

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Early exit for upper triangular part
    if (row < col) {
        if (row < N && col < N) {
            C[row * N + col] = 0.0f;
        }
        return;
    }

    float sum = 0.0f;
    const int start_tile = col / TILE_SIZE;
    
    for (int t = start_tile; t <= blockIdx.y; ++t) {
        // Load tiles into shared memory
        load_tile_to_shared((float*)shared_A, A, 
                          row, t * TILE_SIZE + threadIdx.x,
                          N, threadIdx.y, threadIdx.x);
        load_tile_to_shared((float*)shared_B, B,
                          t * TILE_SIZE + threadIdx.y, col,
                          N, threadIdx.y, threadIdx.x);
        
        __syncthreads();

        if (row < N && col < N) {
            const int k_start = max(t * TILE_SIZE, col);
            const int k_end = min((t + 1) * TILE_SIZE, row + 1);
            
            for (int k = k_start; k < k_end; ++k) {
                const int tile_k = k - t * TILE_SIZE;
                sum += shared_A[threadIdx.y][tile_k] * shared_B[tile_k][threadIdx.x];
            }
        }
        
        // Synchronize only if there are more tiles to process
        if (t < blockIdx.y) {
            __syncthreads();
        }
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
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
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    // Use cudaFuncSetCacheConfig to prefer L1 cache
    cudaFuncSetCacheConfig(triangular_mm_kernel, cudaFuncCachePreferL1);

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}