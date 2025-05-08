#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32
#define WARPS_PER_BLOCK 8
#define STRIDE_FACTOR 4

__device__ __forceinline__ void process_tile(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* sum,
                                            const int row,
                                            const int col,
                                            const int tile_idx,
                                            const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Collaborative loading of tiles using all threads
    #pragma unroll
    for (int i = tid; i < TILE_SIZE * TILE_SIZE; i += blockDim.x * blockDim.y) {
        int load_row = i / TILE_SIZE;
        int load_col = i % TILE_SIZE;
        
        int global_row = row + load_row;
        int global_col = tile_idx * TILE_SIZE + load_col;
        
        if (global_row < N && global_col < N && global_row >= global_col) {
            As[load_row][load_col] = __ldg(&A[global_row * N + global_col]);
        } else {
            As[load_row][load_col] = 0.0f;
        }
        
        global_row = tile_idx * TILE_SIZE + load_row;
        global_col = col + load_col;
        
        if (global_row < N && global_col < N && global_row >= global_col) {
            Bs[load_row][load_col] = __ldg(&B[global_row * N + global_col]);
        } else {
            Bs[load_row][load_col] = 0.0f;
        }
    }
    __syncthreads();
    
    // Process multiple elements per thread using strided access
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        if (tile_idx * TILE_SIZE + k <= row && tile_idx * TILE_SIZE + k >= col) {
            *sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
    }
    __syncthreads();
}

__global__ void strided_triangular_mm_kernel(const float* __restrict__ A,
                                           const float* __restrict__ B,
                                           float* __restrict__ C,
                                           const int N) {
    const int tid = threadIdx.x + threadIdx.y * blockDim.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    
    // Each thread processes multiple elements with stride
    #pragma unroll
    for (int stride = 0; stride < STRIDE_FACTOR; stride++) {
        int row = blockIdx.y * blockDim.y * STRIDE_FACTOR + threadIdx.y + stride * blockDim.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row >= N || col >= N) continue;
        
        if (row < col) {
            C[row * N + col] = 0.0f;
            continue;
        }
        
        float sum = 0.0f;
        const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
        
        // Process tiles
        for (int t = col / TILE_SIZE; t <= row / TILE_SIZE && t < num_tiles; t++) {
            process_tile(A, B, &sum, row, col, t, N);
        }
        
        C[row * N + col] = sum;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Tensors must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimensions must match");
    
    const int N = A.size(0);
    auto C = torch::empty_like(A);
    
    dim3 block(TILE_SIZE, WARPS_PER_BLOCK);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE * STRIDE_FACTOR - 1) / (TILE_SIZE * STRIDE_FACTOR));
    
    strided_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided warp-based triangular matrix multiplication (CUDA)");
}