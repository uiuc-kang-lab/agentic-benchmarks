#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_DIM 32

// Device function to load a tile from matrix A into shared memory
__device__ inline void load_tile_A(const float* __restrict__ A, float tile[TILE_DIM][TILE_DIM],
                                      int M, int K, int tileIdx) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = tileIdx * TILE_DIM + threadIdx.x;
    if (row < M && col < K) 
        tile[threadIdx.y][threadIdx.x] = A[row * K + col];
    else
        tile[threadIdx.y][threadIdx.x] = 0.0f;
}

// Device function to load a tile from matrix B into shared memory
__device__ inline void load_tile_B(const float* __restrict__ B, float tile[TILE_DIM][TILE_DIM],
                                      int K, int N, int tileIdx) {
    int row = tileIdx * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    if (row < K && col < N) 
        tile[threadIdx.y][threadIdx.x] = B[row * N + col];
    else
        tile[threadIdx.y][threadIdx.x] = 0.0f;
}

// Device function to compute the product of the loaded A and B tiles
__device__ inline float compute_tile(const float tileA[TILE_DIM][TILE_DIM],
                                       const float tileB[TILE_DIM][TILE_DIM]) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
        sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
    }
    return sum;
}

// Main kernel that performs matrix multiplication using modular device functions
__global__ void modular_matrix_multiply_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int M, int N, int K) {
    __shared__ float tileA[TILE_DIM][TILE_DIM];
    __shared__ float tileB[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; t++) {
        // Load a tile from A and B into shared memory
        load_tile_A(A, tileA, M, K, t);
        load_tile_B(B, tileB, K, N, t);
        __syncthreads();

        // Compute partial dot product for this tile
        sum += compute_tile(tileA, tileB);
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float *d_A = A.data_ptr<float>();
    const float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    modular_matrix_multiply_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int N = B.size(1);

    torch::Tensor C = torch::empty({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular matrix multiplication (CUDA)");
}
