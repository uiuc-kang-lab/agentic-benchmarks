#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define TILE_SIZE 16
#define MATRIX_SIZE_THRESHOLD 512

// Device function to load a tile of matrix A into shared memory
__device__ void load_tile_A(const float* __restrict__ A, float tileA[TILE_SIZE][TILE_SIZE], int M, int K, int rowStart, int colStart) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = rowStart + ty;
    int col = colStart + tx;
    if (row < M && col < K) {
        tileA[ty][tx] = A[row * K + col];
    } else {
        tileA[ty][tx] = 0.0f;
    }
}

// Device function to load a tile of matrix B into shared memory
__device__ void load_tile_B(const float* __restrict__ B, float tileB[TILE_SIZE][TILE_SIZE], int K, int N, int rowStart, int colStart) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = rowStart + ty;
    int col = colStart + tx;
    if (row < K && col < N) {
        tileB[ty][tx] = B[row * N + col];
    } else {
        tileB[ty][tx] = 0.0f;
    }
}

// Device function to compute partial product for the loaded tile
__device__ float compute_tile_partial(const float tileA[TILE_SIZE][TILE_SIZE], 
                                        const float tileB[TILE_SIZE][TILE_SIZE], 
                                        int tx, int ty, int effectiveWidth) {
    float sum = 0.0f;
    for (int k = 0; k < effectiveWidth; ++k) {
        sum += tileA[ty][k] * tileB[k][tx];
    }
    return sum;
}

// Main kernel using modular device functions
__global__ void ModularMatmulKernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int M, int K, int N) {
    // Determine the starting row and column for this block's tile of C
    int rowStart = blockIdx.y * TILE_SIZE;
    int colStart = blockIdx.x * TILE_SIZE;
    
    // Local thread indices within the tile
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    float cValue = 0.0f;

    // Declare shared memory for tiles of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Loop over the tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int colA = t * TILE_SIZE;  // Starting column in A for this tile
        int rowB = t * TILE_SIZE;  // Starting row in B for this tile

        // Load the tile from A and B into shared memory using modular functions
        load_tile_A(A, tileA, M, K, rowStart, colA);
        load_tile_B(B, tileB, K, N, rowB, colStart);

        __syncthreads();

        // Determine effective tile width (handles partial tiles in the K dimension)
        int effectiveWidth = TILE_SIZE;
        if (t == numTiles - 1 && (K % TILE_SIZE != 0)) {
            effectiveWidth = K % TILE_SIZE;
        }

        // Accumulate the partial product
        cValue += compute_tile_partial(tileA, tileB, tx, ty, effectiveWidth);

        __syncthreads();
    }

    // Write the computed value to C if within bounds
    int row = rowStart + ty;
    int col = colStart + tx;
    if (row < M && col < N) {
        C[row * N + col] = cValue;
    }
}

// Forward function that selects the custom kernel for small matrices
// or falls back to cuBLAS for larger ones.
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    if (M <= MATRIX_SIZE_THRESHOLD && N <= MATRIX_SIZE_THRESHOLD) {
        dim3 blockDim(TILE_SIZE, TILE_SIZE);
        dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        ModularMatmulKernel<<<gridDim, blockDim>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, K, N
        );
    } else {
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
        }
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    B.data_ptr<float>(), N,
                    A.data_ptr<float>(), K,
                    &beta, C.data_ptr<float>(), N);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular tiled matrix multiplication (CUDA)");
}
