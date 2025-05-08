#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 16

template <typename scalar_t>
__global__ void divergence_free_matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    // Calculate thread position
    const int warpId = threadIdx.x / WARP_SIZE;
    const int laneId = threadIdx.x % WARP_SIZE;
    
    // Calculate global position
    const int row = blockIdx.y * BLOCK_SIZE_M + laneId;
    const int col = blockIdx.x * BLOCK_SIZE_N + threadIdx.y;
    
    // Pre-compute validity flags
    const bool valid_row = row < M;
    const bool valid_col = col < N;
    const bool valid_thread = valid_row && valid_col;
    
    // Accumulator
    scalar_t sum = 0;
    
    // Shared memory
    __shared__ scalar_t As[BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ scalar_t Bs[BLOCK_SIZE_N][BLOCK_SIZE_K];
    
    // Process K in tiles
    for (int tile = 0; tile < (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++tile) {
        const int k_start = tile * BLOCK_SIZE_K;
        
        // Collaborative loading of A and B tiles into shared memory
        // Each warp handles a continuous segment of memory
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; k += (BLOCK_SIZE_M * BLOCK_SIZE_N) / (WARP_SIZE * BLOCK_SIZE_K)) {
            const int k_local = k + (threadIdx.x % (BLOCK_SIZE_K / ((BLOCK_SIZE_M * BLOCK_SIZE_N) / (WARP_SIZE * BLOCK_SIZE_K))));
            if (k_local < BLOCK_SIZE_K) {
                const int k_global = k_start + k_local;
                const bool valid_k = k_global < K;
                
                // Load A - use predication instead of branching
                scalar_t a_val = valid_k && valid_row ? __ldg(&A[k_global * M + row]) : 0;
                As[k_local][laneId] = a_val;
                
                // Load B - use predication instead of branching
                scalar_t b_val = valid_k && valid_col ? __ldg(&B[col * K + k_global]) : 0;
                Bs[threadIdx.y][k_local] = b_val;
            }
        }
        
        __syncthreads();
        
        // Compute partial products
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            sum = __fmaf_rn(As[k][laneId], Bs[threadIdx.y][k], sum);
        }
        
        __syncthreads();
    }
    
    // Write result using predication instead of branching
    if (valid_thread) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    // Configure execution parameters
    dim3 threads(WARP_SIZE, BLOCK_SIZE_N/2);  // One warp per row, multiple columns
    dim3 blocks(
        (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
        (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M
    );
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "divergence_free_matmul_transpose_kernel", ([&] {
        divergence_free_matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Divergence-free matrix multiplication with transpose (CUDA)");
}