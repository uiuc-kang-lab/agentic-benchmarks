#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__device__ void load_tile_to_shared(float* shared_tile, 
                                  const float* global_mat,
                                  int row, int col, 
                                  int N, 
                                  int tile_row, int tile_col) {
    if (row < N && col < N) {
        shared_tile[tile_row * TILE_SIZE + tile_col] = global_mat[row * N + col];
    } else {
        shared_tile[tile_row * TILE_SIZE + tile_col] = 0.0f;
    }
}

__device__ float compute_partial_sum(const float* shared_A, 
                                   const float* shared_B,
                                   int tile_row, int tile_col, 
                                   int tile_idx) {
    return shared_A[tile_row * TILE_SIZE + tile_idx] * 
           shared_B[tile_idx * TILE_SIZE + tile_col];
}

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row < col) {
        if (row < N && col < N) {
            C[row * N + col] = 0.0f;
        }
        return;
    }

    float sum = 0.0f;
    
    // Iterate over tiles
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Load tiles into shared memory
        load_tile_to_shared((float*)shared_A, A, 
                          row, t * TILE_SIZE + threadIdx.x,
                          N, threadIdx.y, threadIdx.x);
        load_tile_to_shared((float*)shared_B, B,
                          t * TILE_SIZE + threadIdx.y, col,
                          N, threadIdx.y, threadIdx.x);
        
        __syncthreads();

        // Compute partial sums for this tile
        if (row < N && col < N) {
            int k_start = max(t * TILE_SIZE, col);
            int k_end = min((t + 1) * TILE_SIZE, row + 1);
            
            for (int k = k_start; k < k_end; ++k) {
                int tile_k = k - t * TILE_SIZE;
                sum += compute_partial_sum((float*)shared_A, (float*)shared_B,
                                        threadIdx.y, threadIdx.x, tile_k);
            }
        }
        
        __syncthreads();
    }

    if (row < N && col < N) {
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);

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