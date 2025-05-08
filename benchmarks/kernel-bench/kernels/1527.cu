#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Device function to load a tile from matrix A into shared memory
__device__ inline void loadTileA(const float* __restrict__ A,
                                  float s_A[TILE_SIZE][TILE_SIZE],
                                  int row, int tileIdx, int tx, int ty, int N) {
    int col = tileIdx * TILE_SIZE + tx;
    s_A[ty][tx] = (row < N && col < N) ? A[row * N + col] : 0.0f;
}

// Device function to load a tile from matrix B into shared memory
__device__ inline void loadTileB(const float* __restrict__ B,
                                  float s_B[TILE_SIZE][TILE_SIZE],
                                  int col, int tileIdx, int tx, int ty, int N) {
    int row = tileIdx * TILE_SIZE + ty;
    s_B[ty][tx] = (row < N && col < N) ? B[row * N + col] : 0.0f;
}

// Device function to compute the product of the current tiles
__device__ inline float multiplyTile(const float s_A[TILE_SIZE][TILE_SIZE],
                                       const float s_B[TILE_SIZE][TILE_SIZE],
                                       int tx, int ty) {
    float sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        sum += s_A[ty][k] * s_B[k][tx];
    }
    return sum;
}

// Main CUDA kernel using modular device functions for tiled matrix multiplication
__global__ void matmul_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float value = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load a tile of A and B into shared memory using modular functions
        loadTileA(A, s_A, row, t, tx, ty, N);
        loadTileB(B, s_B, col, t, tx, ty, N);

        __syncthreads();

        // Accumulate the product of the two tiles
        value += multiplyTile(s_A, s_B, tx, ty);

        __syncthreads();
    }

    // Write the computed value to the output matrix if indices are valid
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// C++ interface function
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication with Modular Device Functions (CUDA)");
}
