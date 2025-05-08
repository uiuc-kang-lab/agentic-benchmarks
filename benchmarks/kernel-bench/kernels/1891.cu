#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define CHUNK_SIZE 8
#define MAX_THREADS_PER_BLOCK 1024

__device__ __forceinline__ void load_tile(float (&s_A)[BLOCK_SIZE][BLOCK_SIZE],
                                        float (&s_B)[BLOCK_SIZE][BLOCK_SIZE],
                                        const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        const int row, const int col,
                                        const int block_start, const int N) {
    if (row < N && (block_start + threadIdx.x) < N) {
        s_A[threadIdx.y][threadIdx.x] = A[row * N + block_start + threadIdx.x];
    } else {
        s_A[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    if ((block_start + threadIdx.y) < N && col < N) {
        s_B[threadIdx.y][threadIdx.x] = B[(block_start + threadIdx.y) * N + col];
    } else {
        s_B[threadIdx.y][threadIdx.x] = 0.0f;
    }
}

__device__ __forceinline__ void compute_tile(float (&reg_C)[CHUNK_SIZE],
                                           const float (&s_A)[BLOCK_SIZE][BLOCK_SIZE],
                                           const float (&s_B)[BLOCK_SIZE][BLOCK_SIZE],
                                           const int k_start, const int k_end,
                                           const int block_start) {
    #pragma unroll
    for (int k = k_start; k < k_end; k += CHUNK_SIZE) {
        #pragma unroll
        for (int c = 0; c < CHUNK_SIZE && (k + c) < k_end; c++) {
            reg_C[c] += s_A[threadIdx.y][k - block_start + c] * 
                        s_B[k - block_start + c][threadIdx.x];
        }
    }
}

__device__ __forceinline__ float reduce_registers(float (&reg_C)[CHUNK_SIZE]) {
    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < CHUNK_SIZE; i++) {
        sum += reg_C[i];
    }
    return sum;
}

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];
    
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    
    float reg_C[CHUNK_SIZE] = {0.0f};
    
    // Early exit condition
    if (row >= N || col >= N) return;
    
    // Process tiles
    const int num_tiles = (row / BLOCK_SIZE) + 1;
    
    #pragma unroll 4
    for (int tile = 0; tile <= num_tiles; tile++) {
        const int block_start = tile * BLOCK_SIZE;
        
        // Load tile data
        load_tile(s_A, s_B, A, B, row, col, block_start, N);
        __syncthreads();
        
        // Compute if in lower triangular region
        if (row >= col) {
            const int k_start = max(block_start, col);
            const int k_end = min(block_start + BLOCK_SIZE, row + 1);
            
            compute_tile(reg_C, s_A, s_B, k_start, k_end, block_start);
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row >= col) {
        C[row * N + col] = reduce_registers(reg_C);
    } else {
        C[row * N + col] = 0.0f;
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

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

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