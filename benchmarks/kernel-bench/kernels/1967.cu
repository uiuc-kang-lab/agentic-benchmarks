#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Device function to load a tile of matrix A into shared memory.
__device__ __forceinline__ void load_tile_A(const float* __restrict__ A,
                                              float sA[TILE_SIZE][TILE_SIZE],
                                              int row, int m, int N) {
    int col_index = m * TILE_SIZE + threadIdx.x;
    sA[threadIdx.y][threadIdx.x] = (col_index < N && row >= col_index) ? A[row * N + col_index] : 0.0f;
}

// Device function to load a tile of matrix B into shared memory.
__device__ __forceinline__ void load_tile_B(const float* __restrict__ B,
                                              float sB[TILE_SIZE][TILE_SIZE],
                                              int col, int m, int N) {
    int row_index = m * TILE_SIZE + threadIdx.y;
    sB[threadIdx.y][threadIdx.x] = (row_index < N && row_index >= col) ? B[row_index * N + col] : 0.0f;
}

// Device function to compute the partial sum for the current tile.
__device__ __forceinline__ float compute_tile_sum(const float sA[TILE_SIZE][TILE_SIZE],
                                                    const float sB[TILE_SIZE][TILE_SIZE],
                                                    int col, int row, int m, int N) {
    int tile_start = m * TILE_SIZE;
    int tile_end = tile_start + TILE_SIZE;
    if (tile_end > N) tile_end = N;

    // Determine the valid k range for this tile.
    int kStart = (col > tile_start) ? col : tile_start;
    int kEnd   = ((row + 1) < tile_end) ? (row + 1) : tile_end;

    float tile_sum = 0.0f;
    #pragma unroll
    for (int k = kStart; k < kEnd; k++) {
        int local_k = k - tile_start;
        tile_sum += sA[threadIdx.y][local_k] * sB[local_k][threadIdx.x];
    }
    return tile_sum;
}

// Main kernel that performs lower triangular matrix multiplication using modular device functions.
__global__ void modular_triangular_mm_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;
    
    // Enforce lower triangular result
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int m = 0; m < numTiles; m++) {
        load_tile_A(A, sA, row, m, N);
        load_tile_B(B, sB, col, m, N);
        __syncthreads();
        
        sum += compute_tile_sum(sA, sB, col, row, m, N);
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// PyTorch interface function
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square matrices");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    modular_triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Modular Triangular Matrix Multiplication (CUDA)");
}
