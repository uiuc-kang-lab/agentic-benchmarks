#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define TILE_SIZE 16

// Optimized matrix multiplication kernel leveraging shared memory
__global__ void optimized_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
    // Shared memory for tiles of A and B
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    // Calculate row and column index of C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load data into shared memory
        if (row < M && (t * TILE_SIZE + threadIdx.x) < K) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (t * TILE_SIZE + threadIdx.y) < K) {
            shared_B[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Multiply the two matrices
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += shared_A[threadIdx.y][i] * shared_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result to C if within bounds
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

    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Configure block and grid dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the optimized kernel
    optimized_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Optimized matrix multiplication using shared memory (CUDA)");
}
