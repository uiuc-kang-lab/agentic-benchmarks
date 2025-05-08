#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_WIDTH 16

// Tiled matrix multiplication kernel using 2D thread/block indexing
// Handles irregular shapes by checking bounds.
__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
    // Shared memory for tiles of A and B
    __shared__ float tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float tile_B[TILE_WIDTH][TILE_WIDTH];

    // Compute row and column for C element
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
        int col_A = t * TILE_WIDTH + threadIdx.x;  // Column index for A
        int row_B = t * TILE_WIDTH + threadIdx.y;    // Row index for B

        // Load tile from A (if within bounds)
        if (row < M && col_A < K) {
            tile_A[threadIdx.y][threadIdx.x] = A[row * K + col_A];
        } else {
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B (if within bounds)
        if (row_B < K && col < N) {
            tile_B[threadIdx.y][threadIdx.x] = B[row_B * N + col];
        } else {
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles
        for (int i = 0; i < TILE_WIDTH; i++) {
            sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result to C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function that wraps the CUDA kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure the tensors are CUDA tensors and contiguous
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor C
    auto C = torch::zeros({M, N}, A.options());

    // Define block and grid dimensions using 2D indexing
    dim3 block(TILE_WIDTH, TILE_WIDTH);
    dim3 grid((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    // Launch the CUDA kernel
    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    
    // Ensure kernel completion
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Tiled Matrix Multiplication (CUDA)");
}
