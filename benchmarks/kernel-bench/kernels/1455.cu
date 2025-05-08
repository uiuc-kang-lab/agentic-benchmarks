#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define THREAD_TILE 2
#define MAX_MATRIX_DIM 8192

// Constant memory for matrix dimensions
__constant__ int d_N;

// Optimized kernel using improved block and thread indexing strategies
__global__ void matmul_kernel_thread_optimized(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C) {
    // Calculate row and column for this thread block, considering 2D tile structure
    int blockRow = blockIdx.y * (BLOCK_SIZE / THREAD_TILE) * THREAD_TILE;
    int blockCol = blockIdx.x * (BLOCK_SIZE / THREAD_TILE) * THREAD_TILE;

    // Calculate global indices for the elements computed by this thread
    int row = blockRow + threadIdx.y * THREAD_TILE;
    int col = blockCol + threadIdx.x * THREAD_TILE;

    // Accumulate results for this thread's 2x2 sub-block
    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    // Shared memory allocation for A and B tiles
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    // Iterate over tiles in the k-dimension
    for (int m = 0; m < gridDim.x; ++m) {
        // Load the shared memory sub-matrix for A and B
        int aRow = row;
        int aCol = m * BLOCK_SIZE + threadIdx.x;
        if (aRow < d_N && aCol < d_N) {
            s_A[threadIdx.y][threadIdx.x] = A[aRow * d_N + aCol];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int bRow = m * BLOCK_SIZE + threadIdx.y;
        int bCol = col;
        if (bRow < d_N && bCol < d_N) {
            s_B[threadIdx.y][threadIdx.x] = B[bRow * d_N + bCol];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute this block's contribution to output
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            float a0 = s_A[threadIdx.y * THREAD_TILE + 0][k];
            float a1 = s_A[threadIdx.y * THREAD_TILE + 1][k];
            float b0 = s_B[k][threadIdx.x * THREAD_TILE + 0];
            float b1 = s_B[k][threadIdx.x * THREAD_TILE + 1];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < d_N && col < d_N) C[row * d_N + col] = c00;
    if (row < d_N && (col + 1) < d_N) C[row * d_N + (col + 1)] = c01;
    if ((row + 1) < d_N && col < d_N) C[(row + 1) * d_N + col] = c10;
    if ((row + 1) < d_N && (col + 1) < d_N) C[(row + 1) * d_N + (col + 1)] = c11;
}

// C++ interface (Pybind11 binding)
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);

    // Copy N to constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Configure kernel launch parameters
    dim3 threads(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    matmul_kernel_thread_optimized<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Thread-Optimized Matrix Multiplication (CUDA)");
}