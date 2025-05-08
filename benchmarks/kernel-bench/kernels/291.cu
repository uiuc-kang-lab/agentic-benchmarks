#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 32  // Increased tile size for better occupancy
#define VECTOR_SIZE 4

__device__ __forceinline__ void load_tile_vectorized(
    const float4* __restrict__ src,
    float dst[][TILE],
    const int row,
    const int col,
    const int stride,
    const int max_row,
    const int max_col
) {
    if (row < max_row && col + 3 < max_col) {
        float4 temp = src[row * (stride/4) + col/4];
        dst[threadIdx.y][threadIdx.x*4] = temp.x;
        dst[threadIdx.y][threadIdx.x*4 + 1] = temp.y;
        dst[threadIdx.y][threadIdx.x*4 + 2] = temp.z;
        dst[threadIdx.y][threadIdx.x*4 + 3] = temp.w;
    } else {
        // Handle boundary cases carefully
        dst[threadIdx.y][threadIdx.x*4] = (row < max_row && col < max_col) ? 
            reinterpret_cast<const float*>(src)[row * stride + col] : 0.0f;
        dst[threadIdx.y][threadIdx.x*4 + 1] = (row < max_row && col + 1 < max_col) ? 
            reinterpret_cast<const float*>(src)[row * stride + col + 1] : 0.0f;
        dst[threadIdx.y][threadIdx.x*4 + 2] = (row < max_row && col + 2 < max_col) ? 
            reinterpret_cast<const float*>(src)[row * stride + col + 2] : 0.0f;
        dst[threadIdx.y][threadIdx.x*4 + 3] = (row < max_row && col + 3 < max_col) ? 
            reinterpret_cast<const float*>(src)[row * stride + col + 3] : 0.0f;
    }
}

__global__ void bmm_optimized_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int batch_size,
    const int M,
    const int K,
    const int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * TILE + ty;
    const int col = bx * TILE + tx;

    const int batch_offset_a = bz * M * K;
    const int batch_offset_b = bz * K * N;
    
    float sum = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        load_tile_vectorized(
            (const float4*)(A + batch_offset_a + row * K + t * TILE),
            As,
            ty,
            tx * 4,
            K,
            M,
            K - t * TILE
        );

        load_tile_vectorized(
            (const float4*)(B + batch_offset_b + (t * TILE + ty) * N),
            Bs,
            ty,
            tx * 4,
            N,
            K - t * TILE,
            N
        );

        __syncthreads();

        #pragma unroll 8
        for (int k = 0; k < TILE; k++) {
            sum = __fmaf_rn(As[ty][k], Bs[k][tx], sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[bz * M * N + row * N + col] = sum;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    auto C = torch::zeros({batch_size, M, N}, A.options());

    dim3 threads(TILE/4, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch_size);

    bmm_optimized_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}