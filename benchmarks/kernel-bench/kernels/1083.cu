#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define UNROLL_FACTOR 4

template <typename scalar_t>
__global__ void optimized_hybrid_matmul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int N, const int M, const int K, const int L) {

    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM];

    const int batch_id = blockIdx.z;
    const int row = blockIdx.y * BLOCK_ROWS + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    scalar_t thread_results[BLOCK_ROWS] = {0};
    
    const scalar_t* batch_A = A + batch_id * M * K;
    const scalar_t* batch_B = B;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        #pragma unroll
        for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
            if ((row + i) < M && (t * TILE_DIM + threadIdx.x) < K) {
                tile_A[threadIdx.y + i][threadIdx.x] = 
                    batch_A[(row + i) * K + t * TILE_DIM + threadIdx.x];
            } else {
                tile_A[threadIdx.y + i][threadIdx.x] = 0;
            }
        }

        if ((t * TILE_DIM + threadIdx.y) < K && col < L) {
            tile_B[threadIdx.y][threadIdx.x] = 
                batch_B[(t * TILE_DIM + threadIdx.y) * L + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0;
        }

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; i++) {
            scalar_t sum = 0;
            #pragma unroll
            for (int k = 0; k < TILE_DIM; k += UNROLL_FACTOR) {
                sum += tile_A[threadIdx.y + i][k] * tile_B[k][threadIdx.x];
                sum += tile_A[threadIdx.y + i][k + 1] * tile_B[k + 1][threadIdx.x];
                sum += tile_A[threadIdx.y + i][k + 2] * tile_B[k + 2][threadIdx.x];
                sum += tile_A[threadIdx.y + i][k + 3] * tile_B[k + 3][threadIdx.x];
            }
            thread_results[i] += sum;
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < BLOCK_ROWS; i++) {
        if ((row + i) < M && col < L) {
            output[batch_id * M * L + (row + i) * L + col] = thread_results[i];
        }
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);

    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, 
              (M + BLOCK_ROWS - 1) / BLOCK_ROWS,
              N);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "module_fn_cuda_forward", ([&] {
        optimized_hybrid_matmul_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}