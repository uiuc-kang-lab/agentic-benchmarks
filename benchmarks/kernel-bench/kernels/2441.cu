#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 32
#define BLOCK_SIZE_K 32

template <typename scalar_t>
__global__ void matmul_transpose_aligned_ldg_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {

    // Block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    const int aBegin = K * BLOCK_SIZE_M * by;
    
    // Index of the first sub-matrix of B processed by the block
    const int bBegin = BLOCK_SIZE_N * bx;

    // Shared memory
    __shared__ scalar_t As[BLOCK_SIZE_K][BLOCK_SIZE_M + 1]; // +1 for bank conflicts
    __shared__ scalar_t Bs[BLOCK_SIZE_N][BLOCK_SIZE_K + 1]; // +1 for bank conflicts

    // Accumulator registers
    scalar_t Csub[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    // Loop over all the sub-matrices of A and B
    for (int t = 0; t < (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++t) {
        // Load the matrices from global memory to shared memory
        
        // Each thread loads 4 elements using vector loads when possible
        const int a_row = ty;
        const int a_col = tx;
        const int b_row = ty;
        const int b_col = tx;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int a_global_row = t * BLOCK_SIZE_K + a_row;
            const int a_global_col = by * BLOCK_SIZE_M + a_col + i * 8;
            
            if (a_global_row < K && a_global_col < M) {
                As[a_row][a_col + i * 8] = __ldg(&A[a_global_row * M + a_global_col]);
            } else {
                As[a_row][a_col + i * 8] = 0.0f;
            }

            const int b_global_row = bx * BLOCK_SIZE_N + b_row + i * 8;
            const int b_global_col = t * BLOCK_SIZE_K + b_col;
            
            if (b_global_row < N && b_global_col < K) {
                Bs[b_row + i * 8][b_col] = __ldg(&B[b_global_row * K + b_global_col]);
            } else {
                Bs[b_row + i * 8][b_col] = 0.0f;
            }
        }

        __syncthreads();

        // Multiply the two matrices together
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                scalar_t a_val = As[k][tx + i * 8];
                scalar_t b_val = Bs[ty + i * 8][k];
                Csub[i] += a_val * b_val;
            }
        }

        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const int c_row = by * BLOCK_SIZE_M + tx;
        const int c_col = bx * BLOCK_SIZE_N + ty + i * 8;
        
        if (c_row < M && c_col < N) {
            C[c_row * N + c_col] = Csub[i];
        }
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(8, 8);  // 8x8 = 64 threads per block
    dim3 blocks(
        (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
        (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M
    );

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_aligned_ldg_kernel", ([&] {
        matmul_transpose_aligned_ldg_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transpose (CUDA)");
}