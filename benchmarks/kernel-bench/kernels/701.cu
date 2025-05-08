#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define BLOCK_K 8
#define PAD 1

template <typename scalar_t>
__global__ void matmul_shared_optimized(const scalar_t* __restrict__ A,
                                      const scalar_t* __restrict__ B,
                                      scalar_t* __restrict__ C,
                                      const int M, const int K, const int N) {
    // Shared memory with padding to avoid bank conflicts
    __shared__ scalar_t As[2][TILE_WIDTH][TILE_WIDTH + PAD];
    __shared__ scalar_t Bs[2][TILE_WIDTH][TILE_WIDTH + PAD];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_WIDTH;
    const int by = blockIdx.y * TILE_WIDTH;
    
    const int row = by + ty;
    const int col = bx + tx;
    
    scalar_t sum = 0.0f;
    
    // Registers to cache values from shared memory
    scalar_t rA[BLOCK_K];
    scalar_t rB[BLOCK_K];
    
    // Loop over tiles with double buffering
    int buffer = 0;
    
    // Preload first tile
    if (row < M && tx < K)
        As[0][ty][tx] = A[row * K + tx];
    if (col < N && ty < K)
        Bs[0][ty][tx] = B[ty * N + col];
        
    __syncthreads();
    
    // Main loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load next tile into alternate buffer
        if (t < (K + TILE_WIDTH - 1) / TILE_WIDTH - 1) {
            const int next_t = (t + 1) * TILE_WIDTH;
            if (row < M && next_t + tx < K)
                As[1-buffer][ty][tx] = A[row * K + next_t + tx];
            if (col < N && next_t + ty < K)
                Bs[1-buffer][ty][tx] = B[(next_t + ty) * N + col];
        }
        
        // Compute on current buffer
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; k += BLOCK_K) {
            // Cache shared memory values into registers
            #pragma unroll
            for (int b = 0; b < BLOCK_K; ++b) {
                rA[b] = As[buffer][ty][k + b];
                rB[b] = Bs[buffer][k + b][tx];
            }
            
            // Compute using registered values
            #pragma unroll
            for (int b = 0; b < BLOCK_K; ++b) {
                sum = __fmaf_rn(rA[b], rB[b], sum);
            }
        }
        
        buffer = 1 - buffer;
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    
    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_shared_optimized", ([&] {
        matmul_shared_optimized<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Optimized matrix multiplication (CUDA)");
}