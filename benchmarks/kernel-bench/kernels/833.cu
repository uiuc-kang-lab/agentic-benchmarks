#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

#define TILE_SIZE 16

// CUDA kernel using stride loops to cover multiple output tiles per block
__global__ void matmul_stride_kernel(const float* __restrict__ A, 
                                       const float* __restrict__ B, 
                                       float* __restrict__ C, 
                                       int M, int K, int N) {
    // Declare shared memory for sub-tiles of A and B
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    // Each thread's local indices within a tile
    int local_row = threadIdx.y;
    int local_col = threadIdx.x;

    // Use grid-stride loops to cover the entire output matrix
    for (int row_tile = blockIdx.y * TILE_SIZE; row_tile < M; row_tile += gridDim.y * TILE_SIZE) {
        for (int col_tile = blockIdx.x * TILE_SIZE; col_tile < N; col_tile += gridDim.x * TILE_SIZE) {
            float acc = 0.0f;

            // Loop over the K dimension in tiles
            int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
            for (int t = 0; t < numTiles; t++) {
                int A_col = t * TILE_SIZE + local_col;
                int A_row = row_tile + local_row;
                int B_row = t * TILE_SIZE + local_row;
                int B_col = col_tile + local_col;

                // Load data into shared memory with boundary checks
                shared_A[local_row][local_col] = (A_row < M && A_col < K) ? A[A_row * K + A_col] : 0.0f;
                shared_B[local_row][local_col] = (B_row < K && B_col < N) ? B[B_row * N + B_col] : 0.0f;
                
                __syncthreads();

                // Multiply the two sub-tiles together
                #pragma unroll
                for (int k = 0; k < TILE_SIZE; k++) {
                    acc += shared_A[local_row][k] * shared_B[k][local_col];
                }
                
                __syncthreads();
            }
            
            // Compute the global indices for C
            int c_row = row_tile + local_row;
            int c_col = col_tile + local_col;
            if (c_row < M && c_col < N) {
                C[c_row * N + c_col] = acc;
            }
        }
    }
}

// Host function that initializes and launches the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Allocate the output tensor
    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Define block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    // Use minimal grid covering the matrix; stride loops will cover larger matrices if needed
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    matmul_stride_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    
    // Check for launch errors and synchronize
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Stride Tiled Matrix Multiplication (CUDA)");
}
