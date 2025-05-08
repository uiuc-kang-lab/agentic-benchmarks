#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

#define TILE_SIZE 16

// Tiled matrix multiplication kernel with branchless conditional loads
__global__ void fast_matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int K, int N) {
    // Compute the row and column index of C element
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Allocate shared memory for sub-matrices of A and B
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    // Loop over all tiles
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int A_col = t * TILE_SIZE + threadIdx.x;
        int B_row = t * TILE_SIZE + threadIdx.y;
        
        // Load elements into shared memory using branchless conditional logic
        // Instead of using divergent if-statements, we use the ternary operator
        float a_val = (row < M && A_col < K) ? A[row * K + A_col] : 0.0f;
        float b_val = (B_row < K && col < N) ? B[B_row * N + col] : 0.0f;

        shared_A[threadIdx.y][threadIdx.x] = a_val;
        shared_B[threadIdx.y][threadIdx.x] = b_val;
        __syncthreads();

        // Compute partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result back to C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Wrapper function to launch the CUDA kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Configure the block and grid dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    fast_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Fast matrix multiplication with reduced warp divergence (CUDA)");
}
