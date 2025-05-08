#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>

// Tile size set equal to warp size (32) for best coalescing
#define TILE_SIZE 32

// Kernel that performs tiled matrix multiplication with aligned, coalesced loads
__global__ void vectorized_coalesced_matmul_kernel(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C,
                                                    int M, int K, int N) {
    // Compute global row and column indices for C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Declare shared memory tiles for A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Loop over tiles along the K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Global column index for A tile load
        int aCol = t * TILE_SIZE + threadIdx.x;
        // Global row index for B tile load
        int bRow = t * TILE_SIZE + threadIdx.y;

        // Load A tile: each thread loads one element from A
        // Since A is row-major, threads in a warp (same row) access consecutive memory locations
        if (row < M && aCol < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile: each thread loads one element from B
        // For B, indexing ensures that threads in the same row (threadIdx.y fixed) load consecutive elements
        if (bRow < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two tiles together
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value back to C ensuring coalesced write
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch the CUDA kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate output tensor
    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Setup grid and block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    vectorized_coalesced_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Vectorized and coalesced matrix multiplication (CUDA)");
}
