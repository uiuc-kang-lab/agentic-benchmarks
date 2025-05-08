#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 16
#define BLOCK_ROWS 16

// Device function to load a tile into shared memory
__device__ void load_tile_to_shared(float *dst, const float *src, 
                                   int row, int col, int width, int tile_dim) {
    if (row < width && col < width) {
        dst[threadIdx.y * tile_dim + threadIdx.x] = src[row * width + col];
    } else {
        dst[threadIdx.y * tile_dim + threadIdx.x] = 0.0f;
    }
}

// Device function to compute partial result for a tile
__device__ float compute_tile_result(const float *tile_A, const float *tile_B,
                                   int tile_idx, int row, int col, int width) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_DIM; k++) {
        int global_k = tile_idx * TILE_DIM + k;
        if (global_k <= row) {
            sum += tile_A[threadIdx.y * TILE_DIM + k] * 
                   tile_B[k * TILE_DIM + threadIdx.x];
        }
    }
    return sum;
}

__global__ void triangular_mm_kernel_modular(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           int N) {
    __shared__ float shared_A[TILE_DIM][TILE_DIM];
    __shared__ float shared_B[TILE_DIM][TILE_DIM];
    
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    float sum = 0.0f;
    
    // Iterate over tiles
    for (int tile = 0; tile <= blockIdx.y; tile++) {
        // Load tiles into shared memory
        load_tile_to_shared(&shared_A[0][0], A, 
                          row, tile * TILE_DIM + threadIdx.x,
                          N, TILE_DIM);
        load_tile_to_shared(&shared_B[0][0], B,
                          tile * TILE_DIM + threadIdx.y, col,
                          N, TILE_DIM);
        
        __syncthreads();
        
        // Compute partial results
        sum += compute_tile_result(&shared_A[0][0], &shared_B[0][0],
                                 tile, row, col, N);
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = (row >= col) ? sum : 0.0f;
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

    dim3 threadsPerBlock(TILE_DIM, BLOCK_ROWS);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM,
                   (N + BLOCK_ROWS - 1) / BLOCK_ROWS);

    triangular_mm_kernel_modular<<<numBlocks, threadsPerBlock>>>(
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