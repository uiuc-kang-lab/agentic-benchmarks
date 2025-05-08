#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_SIZE 16
#define THREAD_TILE 2
#define PREFETCH_DISTANCE 2

__device__ __forceinline__ void loadTileAsync(const float* __restrict__ src,
                                            float dst[TILE_DIM][TILE_DIM],
                                            int M, int N, int row_offset, int col_offset,
                                            int threadId, int numThreads) {
    #pragma unroll
    for (int i = threadId; i < TILE_DIM * TILE_DIM; i += numThreads) {
        int localRow = i / TILE_DIM;
        int localCol = i % TILE_DIM;
        int globalRow = row_offset + localRow;
        int globalCol = col_offset + localCol;
        dst[localRow][localCol] = (globalRow < M && globalCol < N) ? 
                                 src[globalRow * N + globalCol] : 0.0f;
    }
}

__device__ __forceinline__ void computeSubTile(float& c00, float& c01, 
                                              float& c10, float& c11,
                                              const float As[TILE_DIM][TILE_DIM],
                                              const float Bs[TILE_DIM][TILE_DIM],
                                              int k, int ty, int tx) {
    float a0 = As[ty * THREAD_TILE + 0][k];
    float a1 = As[ty * THREAD_TILE + 1][k];
    float b0 = Bs[k][tx * THREAD_TILE + 0];
    float b1 = Bs[k][tx * THREAD_TILE + 1];
    
    #pragma unroll
    c00 = __fmaf_rn(a0, b0, c00);
    c01 = __fmaf_rn(a0, b1, c01);
    c10 = __fmaf_rn(a1, b0, c10);
    c11 = __fmaf_rn(a1, b1, c11);
}

__global__ void matmul_kernel(const float* __restrict__ A,
                            const float* __restrict__ B,
                            float* __restrict__ C,
                            int M, int K, int N) {
    __shared__ float As[2][TILE_DIM][TILE_DIM];
    __shared__ float Bs[2][TILE_DIM][TILE_DIM];

    int blockRow = blockIdx.y * TILE_DIM;
    int blockCol = blockIdx.x * TILE_DIM;
    int row = blockRow + threadIdx.y * THREAD_TILE;
    int col = blockCol + threadIdx.x * THREAD_TILE;

    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;
    int numThreads = BLOCK_SIZE * BLOCK_SIZE;
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    
    int current_buffer = 0;
    
    loadTileAsync(A, As[0], M, K, blockRow, 0, threadId, numThreads);
    loadTileAsync(B, Bs[0], K, N, 0, blockCol, threadId, numThreads);
    __syncthreads();

    #pragma unroll 4
    for (int tileIdx = 0; tileIdx < numTiles - 1; tileIdx++) {
        loadTileAsync(A, As[1-current_buffer], M, K, 
                     blockRow, (tileIdx + 1) * TILE_DIM,
                     threadId, numThreads);
        loadTileAsync(B, Bs[1-current_buffer], K, N,
                     (tileIdx + 1) * TILE_DIM, blockCol,
                     threadId, numThreads);

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            computeSubTile(c00, c01, c10, c11,
                          As[current_buffer], Bs[current_buffer],
                          k, threadIdx.y, threadIdx.x);
        }
        
        current_buffer = 1 - current_buffer;
        __syncthreads();
    }

    #pragma unroll
    for (int k = 0; k < TILE_DIM; k++) {
        computeSubTile(c00, c01, c10, c11,
                      As[current_buffer], Bs[current_buffer],
                      k, threadIdx.y, threadIdx.x);
    }

    if (row < M && col < N) {
        if ((col + 1) < N && (row + 1) < M) {
            float2* row0 = (float2*)&C[row * N + col];
            float2* row1 = (float2*)&C[(row + 1) * N + col];
            *row0 = make_float2(c00, c01);
            *row1 = make_float2(c10, c11);
        } else {
            C[row * N + col] = c00;
            if ((col + 1) < N) C[row * N + col + 1] = c01;
            if ((row + 1) < M) {
                C[(row + 1) * N + col] = c10;
                if ((col + 1) < N) C[(row + 1) * N + col + 1] = c11;
            }
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    return C;
}