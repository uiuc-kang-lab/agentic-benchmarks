#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define MAX_MATRIX_DIM 8192

// Kernel using grid-stride loops to cover the entire output matrix
// Each block processes potentially more than one tile via row and column striding
__global__ void matmul_kernel_stride(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    // Declare shared memory for tiling
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    // Loop over output tiles for rows and columns using grid-stride loops
    for (int row_tile = blockIdx.y * BLOCK_SIZE; row_tile < N; row_tile += gridDim.y * BLOCK_SIZE) {
        for (int col_tile = blockIdx.x * BLOCK_SIZE; col_tile < N; col_tile += gridDim.x * BLOCK_SIZE) {
            // Compute the global row and column for each thread within the current tile
            int row = row_tile + threadIdx.y;
            int col = col_tile + threadIdx.x;
            float sum = 0.0f;
            
            // Loop over tiles in the k dimension
            for (int t = 0; t < N; t += BLOCK_SIZE) {
                // Load tile from matrix A into shared memory with boundary check
                int a_row = row;
                int a_col = t + threadIdx.x;
                if (a_row < N && a_col < N)
                    s_A[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
                else
                    s_A[threadIdx.y][threadIdx.x] = 0.0f;

                // Load tile from matrix B into shared memory with boundary check
                int b_row = t + threadIdx.y;
                int b_col = col;
                if (b_row < N && b_col < N)
                    s_B[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
                else
                    s_B[threadIdx.y][threadIdx.x] = 0.0f;

                __syncthreads();

                // Compute the partial dot product for the current tile
                for (int k = 0; k < BLOCK_SIZE; k++) {
                    sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
                }

                __syncthreads();
            }
            
            // Write the computed element back to global memory, if in bounds
            if (row < N && col < N) {
                C[row * N + col] = sum;
            }
        }
    }
}

// C++ interface using Pybind11
// This function validates input dimensions and launches the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");
    
    int N = A.size(0);
    TORCH_CHECK(N <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Set thread block size
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    // Launch grid with minimal blocks; kernel uses grid-stride loops to cover full matrix
    int grid_x = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int grid_y = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blocks(grid_x, grid_y);

    matmul_kernel_stride<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided Loop Tiled Matrix Multiplication (CUDA)");
}
