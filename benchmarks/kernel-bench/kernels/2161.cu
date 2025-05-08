#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// This kernel computes C = tril(A * B) for lower triangular matrices A and B
// using shared memory tiling and loop unrolling for full tiles.
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       const int N) {
    __shared__ float shA[TILE_SIZE][TILE_SIZE];
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Check boundaries and triangular condition
    if (row >= N || col >= N) return;
    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    float sum = 0.f;

    // Compute the range of tiles along the k-dimension. For lower triangular matrices,
    // only values from k = col to k = row contribute.
    int t_start = col / TILE_SIZE;
    int t_end   = row / TILE_SIZE;

    for (int t = t_start; t <= t_end; t++) {
        // Load tile from A. For A[row][k], only k <= row contains valid data.
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (a_col < N && a_col <= row) {
            shA[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        } else {
            shA[threadIdx.y][threadIdx.x] = 0.f;
        }
        
        // Load tile from B. For B[k][col], only k >= col contains nonzero values.
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < N && b_row >= col) {
            shB[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            shB[threadIdx.y][threadIdx.x] = 0.f;
        }
        
        __syncthreads();
        
        // Determine effective k range in this tile: intersect [t*TILE_SIZE, (t+1)*TILE_SIZE) with [col, row+1).
        int k_tile_start = t * TILE_SIZE;
        int k_tile_end = (t + 1) * TILE_SIZE;
        int k_begin = (k_tile_start > col) ? k_tile_start : col;
        int k_end = (k_tile_end < row + 1) ? k_tile_end : row + 1;
        int valid_elements = k_end - k_begin;

        // If the entire tile is valid, unroll the loop for maximum performance
        if (valid_elements == TILE_SIZE) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += shA[threadIdx.y][k] * shB[k][threadIdx.x];
            }
        } else {
            for (int k = k_begin; k < k_end; k++) {
                int local_k = k - t * TILE_SIZE;
                sum += shA[threadIdx.y][local_k] * shB[local_k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square matrices");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Efficient lower triangular matrix multiplication (CUDA) using shared memory tiling");
}
