#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define CHUNK_SIZE 8

// Constant memory for frequently accessed parameters
__constant__ int d_N;
__constant__ int d_block_count;
__constant__ float d_zero = 0.0f;

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];
    
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int tid = threadIdx.y * BLOCK_SIZE + threadIdx.x;
    
    // Register array for accumulation
    float reg_C[CHUNK_SIZE] = {0.0f};
    
    // Calculate number of tiles needed for this thread
    const int num_tiles = (min(row + 1, d_N) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Loop over block-level tiles
    for (int tile = 0; tile < num_tiles; tile++) {
        const int block_start = tile * BLOCK_SIZE;
        
        // Collaborative loading with vectorized memory access
        #pragma unroll 2
        for (int i = 0; i < BLOCK_SIZE; i += BLOCK_SIZE/2) {
            const int shared_idx = tid + i;
            const int y = shared_idx / BLOCK_SIZE;
            const int x = shared_idx % BLOCK_SIZE;
            
            if (row < d_N && (block_start + x) < d_N) {
                s_A[y][x] = A[row * d_N + block_start + x];
            } else {
                s_A[y][x] = d_zero;
            }
            
            if ((block_start + y) < d_N && col < d_N) {
                s_B[y][x] = B[(block_start + y) * d_N + col];
            } else {
                s_B[y][x] = d_zero;
            }
        }
        
        __syncthreads();
        
        // Register-level tiling for computation
        if (row < d_N && col < d_N && row >= col) {
            const int k_start = max(block_start, col);
            const int k_end = min(block_start + BLOCK_SIZE, row + 1);
            
            #pragma unroll
            for (int k = k_start; k < k_end; k += CHUNK_SIZE) {
                #pragma unroll
                for (int c = 0; c < CHUNK_SIZE && (k + c) < k_end; c++) {
                    reg_C[c] += s_A[threadIdx.y][k - block_start + c] * 
                               s_B[k - block_start + c][threadIdx.x];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Reduction and writing results
    if (row < d_N && col < d_N) {
        if (row >= col) {
            float sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < CHUNK_SIZE; i++) {
                sum += reg_C[i];
            }
            C[row * d_N + col] = sum;
        } else {
            C[row * d_N + col] = d_zero;
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
    const int block_count = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Copy constants to constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_block_count, &block_count, sizeof(int));

    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(block_count, block_count);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>()
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}