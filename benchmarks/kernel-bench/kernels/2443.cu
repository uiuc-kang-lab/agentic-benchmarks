#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE_M 32
#define BLOCK_SIZE_N 16
#define BLOCK_SIZE_K 16
#define VEC_SIZE 4  // Using float4 for vectorized loads

template <typename scalar_t>
__global__ void matmul_transpose_vectorized_ldg_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {

    // Block index
    const int by = blockIdx.y;
    const int bx = blockIdx.x;

    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Index of the first sub-matrix of A processed by the block
    const int block_row = by * BLOCK_SIZE_M;
    const int block_col = bx * BLOCK_SIZE_N;

    // Shared memory declaration
    __shared__ scalar_t As[BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ scalar_t Bs[BLOCK_SIZE_N][BLOCK_SIZE_K];

    // Accumulator registers
    scalar_t acc[2] = {0.0f, 0.0f};  // Each thread computes 2 output elements

    // Calculate global row indices for this thread
    const int row0 = block_row + tx;
    const int row1 = row0 + BLOCK_SIZE_M/2;
    const int col = block_col + ty;

    // Thread ID in the block
    const int tid = ty * blockDim.x + tx;
    const int block_size = blockDim.x * blockDim.y;

    // Process tiles
    for (int tile = 0; tile < (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++tile) {
        // Load A tile - vectorized when possible
        for (int i = tid; i < (BLOCK_SIZE_K * BLOCK_SIZE_M) / VEC_SIZE; i += block_size) {
            float4* src_ptr = (float4*)&A[tile * BLOCK_SIZE_K * M + (i * VEC_SIZE)];
            if ((tile * BLOCK_SIZE_K + (i * VEC_SIZE) / BLOCK_SIZE_M) < K) {
                float4 tmp = __ldg(src_ptr);
                int k_idx = (i * VEC_SIZE) / BLOCK_SIZE_M;
                int m_idx = (i * VEC_SIZE) % BLOCK_SIZE_M;
                
                // Unpack vector into shared memory
                As[k_idx][m_idx] = tmp.x;
                if (m_idx + 1 < BLOCK_SIZE_M) As[k_idx][m_idx + 1] = tmp.y;
                if (m_idx + 2 < BLOCK_SIZE_M) As[k_idx][m_idx + 2] = tmp.z;
                if (m_idx + 3 < BLOCK_SIZE_M) As[k_idx][m_idx + 3] = tmp.w;
            }
        }

        // Load B tile - vectorized when possible
        for (int i = tid; i < (BLOCK_SIZE_N * BLOCK_SIZE_K) / VEC_SIZE; i += block_size) {
            float4* src_ptr = (float4*)&B[block_col * K + tile * BLOCK_SIZE_K + (i * VEC_SIZE)];
            if ((block_col + (i * VEC_SIZE) / BLOCK_SIZE_K) < N) {
                float4 tmp = __ldg(src_ptr);
                int n_idx = (i * VEC_SIZE) / BLOCK_SIZE_K;
                int k_idx = (i * VEC_SIZE) % BLOCK_SIZE_K;
                
                // Unpack vector into shared memory
                Bs[n_idx][k_idx] = tmp.x;
                if (k_idx + 1 < BLOCK_SIZE_K) Bs[n_idx][k_idx + 1] = tmp.y;
                if (k_idx + 2 < BLOCK_SIZE_K) Bs[n_idx][k_idx + 2] = tmp.z;
                if (k_idx + 3 < BLOCK_SIZE_K) Bs[n_idx][k_idx + 3] = tmp.w;
            }
        }

        __syncthreads();

        // Compute partial products
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            scalar_t a_val0 = As[k][tx];
            scalar_t a_val1 = As[k][tx + BLOCK_SIZE_M/2];
            scalar_t b_val = Bs[ty][k];
            
            acc[0] += a_val0 * b_val;
            acc[1] += a_val1 * b_val;
        }

        __syncthreads();
    }

    // Store results
    if (row0 < M && col < N) {
        C[row0 * N + col] = acc[0];
    }
    if (row1 < M && col < N) {
        C[row1 * N + col] = acc[1];
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(16, 16);
    dim3 blocks((N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N,
                (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_vectorized_ldg_kernel", ([&] {
        matmul_transpose_vectorized_ldg_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transpose using vectorized loads (CUDA)");
}