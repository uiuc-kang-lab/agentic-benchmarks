#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Device function to load a tile of matrix A into shared memory
__device__ __forceinline__ void load_shared_tile_A(const float* __restrict__ A, 
                                                     float sA[TILE_SIZE][TILE_SIZE], 
                                                     int row, int m, int N) {
    int col_idx = m * TILE_SIZE + threadIdx.x;
    if (col_idx < N) {
        sA[threadIdx.y][threadIdx.x] = A[row * N + col_idx];
    } else {
        sA[threadIdx.y][threadIdx.x] = 0.f;
    }
}

// Device function to load a tile of matrix B into shared memory
__device__ __forceinline__ void load_shared_tile_B(const float* __restrict__ B, 
                                                     float sB[TILE_SIZE][TILE_SIZE], 
                                                     int m, int N, int col) {
    int row_idx = m * TILE_SIZE + threadIdx.y;
    if (row_idx < N) {
        sB[threadIdx.y][threadIdx.x] = B[row_idx * N + col];
    } else {
        sB[threadIdx.y][threadIdx.x] = 0.f;
    }
}

// Device function to compute the partial sum for the current tile
// The summation index k corresponds to global k = m * TILE_SIZE + k_local
// Valid k indices must satisfy: k in [col, row] for lower triangular matrices.
__device__ __forceinline__ float compute_tile_sum(const float sA[TILE_SIZE][TILE_SIZE],
                                                     const float sB[TILE_SIZE][TILE_SIZE],
                                                     int row, int col, int m) {
    float sum = 0.f;
    int global_k_start = m * TILE_SIZE;
    int local_start = (col > global_k_start) ? (col - global_k_start) : 0;
    int global_k_end = global_k_start + TILE_SIZE;
    int local_end = (row + 1 < global_k_end) ? (row + 1 - global_k_start) : TILE_SIZE;
    for (int k = local_start; k < local_end; ++k) {
        sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
    }
    return sum;
}

// Modular kernel for lower triangular matrix multiplication using tiled shared memory
// Computes C = A * B for lower triangular matrices, where only elements with row >= col are computed
__global__ void triangular_mm_kernel_modular(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row < N && col < N) {
        // For upper triangular part, output is zero
        if (row < col) {
            C[row * N + col] = 0.f;
            return;
        }

        float sum = 0.f;

        // Allocate shared memory for tiles of A and B
        __shared__ float sA[TILE_SIZE][TILE_SIZE];
        __shared__ float sB[TILE_SIZE][TILE_SIZE];

        int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
        for (int m = 0; m < numTiles; ++m) {
            // Load tiles from global memory into shared memory using modular device functions
            load_shared_tile_A(A, sA, row, m, N);
            load_shared_tile_B(B, sB, m, N, col);
            __syncthreads();

            // Compute the partial sum for this tile
            sum += compute_tile_sum(sA, sB, row, col, m);
            __syncthreads();
        }

        C[row * N + col] = sum;
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel_modular<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular Lower Triangular Matrix Multiplication (CUDA)");
}
