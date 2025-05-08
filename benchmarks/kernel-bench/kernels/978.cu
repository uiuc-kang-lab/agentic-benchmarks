#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define ELEMENTS_PER_THREAD 4
#define TILE_FACTOR 2
#define TILE_DIM (BLOCK_SIZE * TILE_FACTOR)

// Helper to fetch matrix elements considering transpose
__device__ inline float get_element(const float* __restrict__ matrix, int row, int col, int ld, bool transpose) {
    return transpose ? matrix[col * ld + row] : matrix[row * ld + col];
}

__global__ void optimized_tiled_gemm(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int N, int K,
                                      int lda, int ldb, int ldc,
                                      bool transA, bool transB) {
    // Determine the starting row and column for the block
    int blockRow = blockIdx.y * TILE_DIM;
    int blockCol = blockIdx.x * TILE_DIM;

    int threadRow = threadIdx.y;
    int threadCol = threadIdx.x;
    int row0 = blockRow + threadRow * TILE_FACTOR;
    int col0 = blockCol + threadCol * TILE_FACTOR;

    float acc[TILE_FACTOR][TILE_FACTOR] = { {0.0f, 0.0f}, {0.0f, 0.0f} };

    __shared__ float As[TILE_DIM][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][TILE_DIM];

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        int tiledK = t * BLOCK_SIZE;

        // Load a tile of matrix A into shared memory
        for (int i = 0; i < TILE_FACTOR; ++i) {
            int globalRow = row0 + i;
            int globalCol = tiledK + threadIdx.x;
            if (globalRow < M && globalCol < K)
                As[threadIdx.y * TILE_FACTOR + i][threadIdx.x] = get_element(A, globalRow, globalCol, lda, transA);
            else
                As[threadIdx.y * TILE_FACTOR + i][threadIdx.x] = 0.0f;
        }

        // Load a tile of matrix B into shared memory
        for (int i = 0; i < TILE_FACTOR; ++i) {
            int globalRow = tiledK + threadIdx.y;
            int globalCol = col0 + i;
            if (globalRow < K && globalCol < N)
                Bs[threadIdx.y][threadIdx.x * TILE_FACTOR + i] = get_element(B, globalRow, globalCol, ldb, transB);
            else
                Bs[threadIdx.y][threadIdx.x * TILE_FACTOR + i] = 0.0f;
        }

        __syncthreads();

        // Multiply the loaded tiles
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a0 = As[threadIdx.y * TILE_FACTOR + 0][k];
            float a1 = As[threadIdx.y * TILE_FACTOR + 1][k];
            float b0 = Bs[k][threadIdx.x * TILE_FACTOR + 0];
            float b1 = Bs[k][threadIdx.x * TILE_FACTOR + 1];
            acc[0][0] += a0 * b0;
            acc[0][1] += a0 * b1;
            acc[1][0] += a1 * b0;
            acc[1][1] += a1 * b1;
        }

        __syncthreads();
    }

    for (int i = 0; i < TILE_FACTOR; ++i) {
        for (int j = 0; j < TILE_FACTOR; ++j) {
            int globalRow = row0 + i;
            int globalCol = col0 + j;
            if (globalRow < M && globalCol < N)
                C[globalRow * ldc + globalCol] = acc[i][j];
        }
    }
}

torch::Tensor matmul_cuda_optimized(torch::Tensor A, torch::Tensor B) {
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);

    bool transA = false;
    bool transB = false;
    int lda = A.stride(0), ldb = B.stride(0), ldc = N;

    auto C = torch::empty({M, N}, A.options());

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    optimized_tiled_gemm<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc,
        transA, transB);
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda_optimized, "Optimized matrix multiplication combining multiple techniques (CUDA)");
}