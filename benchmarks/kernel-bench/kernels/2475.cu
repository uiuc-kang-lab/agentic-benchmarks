#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__device__ __forceinline__ void load_tile_A_vectorized(
    scalar_t (&tileA)[32][32],
    const scalar_t* __restrict__ A,
    const int row,
    const int tile_idx,
    const int M,
    const int K) {
    
    using float4_t = typename std::conditional<std::is_same<scalar_t, float>::value, float4, double4>::type;
    
    const int vec_size = sizeof(float4_t) / sizeof(scalar_t);
    const scalar_t* row_ptr = A + (tile_idx * 32 + threadIdx.y) * M + (row & ~(vec_size - 1));
    
    if ((tile_idx * 32 + threadIdx.y) < K && row < M) {
        if (row % vec_size == 0 && row + vec_size <= M) {
            float4_t val = *reinterpret_cast<const float4_t*>(row_ptr);
            tileA[threadIdx.y][threadIdx.x] = reinterpret_cast<scalar_t*>(&val)[threadIdx.x % vec_size];
        } else {
            tileA[threadIdx.y][threadIdx.x] = A[(tile_idx * 32 + threadIdx.y) * M + row];
        }
    } else {
        tileA[threadIdx.y][threadIdx.x] = 0.0;
    }
}

template <typename scalar_t>
__device__ __forceinline__ void load_tile_B_vectorized(
    scalar_t (&tileB)[32][32],
    const scalar_t* __restrict__ B,
    const int col,
    const int tile_idx,
    const int N,
    const int K) {
    
    using float4_t = typename std::conditional<std::is_same<scalar_t, float>::value, float4, double4>::type;
    
    const int vec_size = sizeof(float4_t) / sizeof(scalar_t);
    const scalar_t* col_ptr = B + col * K + tile_idx * 32 + (threadIdx.x & ~(vec_size - 1));
    
    if (tile_idx * 32 + threadIdx.x < K && col < N) {
        if (threadIdx.x % vec_size == 0 && threadIdx.x + vec_size <= 32) {
            float4_t val = *reinterpret_cast<const float4_t*>(col_ptr);
            tileB[threadIdx.y][threadIdx.x] = reinterpret_cast<scalar_t*>(&val)[0];
        } else {
            tileB[threadIdx.y][threadIdx.x] = B[col * K + tile_idx * 32 + threadIdx.x];
        }
    } else {
        tileB[threadIdx.y][threadIdx.x] = 0.0;
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_tile_product(
    const scalar_t (&tileA)[32][32],
    const scalar_t (&tileB)[32][32]) {
    
    scalar_t sum = 0;
    #pragma unroll
    for (int k = 0; k < 32; k += 8) {
        sum += tileA[k][threadIdx.x] * tileB[threadIdx.y][k];
        sum += tileA[k+1][threadIdx.x] * tileB[threadIdx.y][k+1];
        sum += tileA[k+2][threadIdx.x] * tileB[threadIdx.y][k+2];
        sum += tileA[k+3][threadIdx.x] * tileB[threadIdx.y][k+3];
        sum += tileA[k+4][threadIdx.x] * tileB[threadIdx.y][k+4];
        sum += tileA[k+5][threadIdx.x] * tileB[threadIdx.y][k+5];
        sum += tileA[k+6][threadIdx.x] * tileB[threadIdx.y][k+6];
        sum += tileA[k+7][threadIdx.x] * tileB[threadIdx.y][k+7];
    }
    return sum;
}

template <typename scalar_t>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ scalar_t tileA[32][32];
    __shared__ scalar_t tileB[32][32];

    scalar_t sum = 0;
    
    for (int t = 0; t < (K + 32 - 1) / 32; ++t) {
        load_tile_A_vectorized(tileA, A, row, t, M, K);
        load_tile_B_vectorized(tileB, B, col, t, N, K);
        
        __syncthreads();
        
        sum += compute_tile_product(tileA, tileB);
        
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    const int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transpose forward (CUDA)");
}