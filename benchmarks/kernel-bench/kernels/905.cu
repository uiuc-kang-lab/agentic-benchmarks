#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <iostream>

// Tiling parameters for custom kernel
#define TILE_DIM 32
#define BLOCK_SIZE 16
#define THREAD_TILE 2

// Device function to load a tile of matrix A into shared memory
__device__ void loadTileA(const float* __restrict__ A, float As[TILE_DIM][TILE_DIM], int M, int K, int tileIdx, int threadId, int blockRow) {
    const int numThreads = BLOCK_SIZE * BLOCK_SIZE;
    for (int i = threadId; i < TILE_DIM * TILE_DIM; i += numThreads) {
        int localRow = i / TILE_DIM;
        int localCol = i % TILE_DIM;
        int globalRow = blockRow + localRow;
        int globalCol = tileIdx * TILE_DIM + localCol;
        if (globalRow < M && globalCol < K)
            As[localRow][localCol] = A[globalRow * K + globalCol];
        else
            As[localRow][localCol] = 0.0f;
    }
}

// Device function to load a tile of matrix B into shared memory
__device__ void loadTileB(const float* __restrict__ B, float Bs[TILE_DIM][TILE_DIM], int K, int N, int tileIdx, int threadId, int blockCol) {
    const int numThreads = BLOCK_SIZE * BLOCK_SIZE;
    for (int i = threadId; i < TILE_DIM * TILE_DIM; i += numThreads) {
        int localRow = i / TILE_DIM;
        int localCol = i % TILE_DIM;
        int globalRow = tileIdx * TILE_DIM + localRow;
        int globalCol = blockCol + localCol;
        if (globalRow < K && globalCol < N)
            Bs[localRow][localCol] = B[globalRow * N + globalCol];
        else
            Bs[localRow][localCol] = 0.0f;
    }
}

// Device function to compute a 2x2 sub-tile from the loaded shared memory tiles
__device__ void computeSubTile(int k, int ty, int tx, float As[TILE_DIM][TILE_DIM], float Bs[TILE_DIM][TILE_DIM], 
                                float &c00, float &c01, float &c10, float &c11) {
    float a0 = As[ty * THREAD_TILE + 0][k];
    float a1 = As[ty * THREAD_TILE + 1][k];
    float b0 = Bs[k][tx * THREAD_TILE + 0];
    float b1 = Bs[k][tx * THREAD_TILE + 1];
    c00 += a0 * b0;
    c01 += a0 * b1;
    c10 += a1 * b0;
    c11 += a1 * b1;
}

// Custom matrix multiplication kernel using shared memory tiling and 2x2 sub-tiles
__global__ void custom_matmul_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int K, int N) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int blockRow = blockIdx.y * TILE_DIM;
    int blockCol = blockIdx.x * TILE_DIM;
    int row = blockRow + threadIdx.y * THREAD_TILE;
    int col = blockCol + threadIdx.x * THREAD_TILE;

    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    int threadId = threadIdx.y * blockDim.x + threadIdx.x;

    for (int tileIdx = 0; tileIdx < numTiles; tileIdx++) {
        loadTileA(A, As, M, K, tileIdx, threadId, blockRow);
        loadTileB(B, Bs, K, N, tileIdx, threadId, blockCol);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            computeSubTile(k, threadIdx.y, threadIdx.x, As, Bs, c00, c01, c10, c11);
        }

        __syncthreads();
    }

    // Write the computed 2x2 sub-tile into matrix C
    if (row < M && col < N)
        C[row * N + col] = c00;
    if (row < M && (col + 1) < N)
        C[row * N + col + 1] = c01;
    if ((row + 1) < M && col < N)
        C[(row + 1) * N + col] = c10;
    if ((row + 1) < M && (col + 1) < N)
        C[(row + 1) * N + col + 1] = c11;
}

// Hybrid matrix multiplication: selects between custom tiled kernel, concurrent custom kernels, or cuBLAS
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Input validation
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Thresholds for different execution strategies
    const int SMALL_THRESHOLD = 256;
    const int CONCURRENT_THRESHOLD = 1024;
    
    // For small matrices: use custom kernel
    // For medium matrices: use concurrent custom kernels
    // For large matrices: use cuBLAS
    if (M <= SMALL_THRESHOLD && N <= SMALL_THRESHOLD && K <= SMALL_THRESHOLD) {
        // Small matrix case: use single custom kernel
        dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
        dim3 blocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
        custom_matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    } 
    else if (M <= CONCURRENT_THRESHOLD && N <= CONCURRENT_THRESHOLD && K <= CONCURRENT_THRESHOLD) {
        // Medium matrix case: split into concurrent custom kernels
        const int CHUNK_SIZE = SMALL_THRESHOLD;
        cudaStream_t streams[4];
        for (int i = 0; i < 4; i++) {
            cudaStreamCreate(&streams[i]);
        }

        for (int m = 0; m < M; m += CHUNK_SIZE) {
            for (int n = 0; n < N; n += CHUNK_SIZE) {
                int current_M = std::min(CHUNK_SIZE, M - m);
                int current_N = std::min(CHUNK_SIZE, N - n);
                
                dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
                dim3 blocks((current_N + TILE_DIM - 1) / TILE_DIM, (current_M + TILE_DIM - 1) / TILE_DIM);
                
                // Use different streams for different chunks to enable concurrent execution
                int streamIdx = ((m/CHUNK_SIZE) + (n/CHUNK_SIZE)) % 4;
                custom_matmul_kernel<<<blocks, threads, 0, streams[streamIdx]>>>(
                    A.data_ptr<float>() + m * K,
                    B.data_ptr<float>() + n,
                    C.data_ptr<float>() + m * N + n,
                    current_M, K, current_N
                );
            }
        }

        // Cleanup streams
        for (int i = 0; i < 4; i++) {
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);
        }
    }
    else {
        // Large matrix case: use cuBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                    B.data_ptr<float>(), N, A.data_ptr<float>(), K, &beta,
                    C.data_ptr<float>(), N);
        cublasDestroy(handle);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Hybrid Matrix Multiplication (CUDA) using cuBLAS and custom kernel");
}
