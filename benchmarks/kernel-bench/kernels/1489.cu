#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size for tiling
#define BLOCK_SIZE 32

// Kernel using grid-stride loops with shared memory tiling
__global__ void matmul_stride_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int N) {
    // Shared memory tiles for A and B
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    // Compute initial output indices based on block and thread indices
    int initial_row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int initial_col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Grid strides in row and column dimensions
    int stride_row = gridDim.y * BLOCK_SIZE;
    int stride_col = gridDim.x * BLOCK_SIZE;

    // Loop over output matrix elements handled by this thread
    for (int row = initial_row; row < N; row += stride_row) {
        for (int col = initial_col; col < N; col += stride_col) {
            float value = 0.0f;
            // Loop over tiles of the input matrices
            int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for (int tile = 0; tile < numTiles; tile++) {
                // Compute global indices for loading data into shared memory
                int A_col = tile * BLOCK_SIZE + threadIdx.x;
                int B_row = tile * BLOCK_SIZE + threadIdx.y;

                // Load element of A into shared memory if within bounds
                if (row < N && A_col < N)
                    s_A[threadIdx.y][threadIdx.x] = A[row * N + A_col];
                else
                    s_A[threadIdx.y][threadIdx.x] = 0.0f;

                // Load element of B into shared memory if within bounds
                if (B_row < N && col < N)
                    s_B[threadIdx.y][threadIdx.x] = B[B_row * N + col];
                else
                    s_B[threadIdx.y][threadIdx.x] = 0.0f;

                __syncthreads();

                // Compute partial product for this tile
                #pragma unroll
                for (int k = 0; k < BLOCK_SIZE; k++) {
                    value += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
                }

                __syncthreads();
            }
            // Write the computed value to the output matrix
            if(row < N && col < N)
                C[row * N + col] = value;
        }
    }
}

// C++ interface function
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Configure launch parameters
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    // Use minimal grid size; grid-stride loops allow handling larger matrices
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    matmul_stride_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication (CUDA) using grid-stride loops");
}
