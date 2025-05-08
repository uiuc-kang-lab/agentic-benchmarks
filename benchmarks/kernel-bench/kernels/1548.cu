#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define the block size to experiment with different occupancy. Using 16 gives 16x16 = 256 threads per block.
// You can modify BLOCK_SIZE to 8, 16, 32 etc. to experiment with performance.
#define BLOCK_SIZE 32

// CUDA kernel for matrix multiplication using shared memory tiling
__global__ void matmul_kernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int N) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    // Compute global row and column indices
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float value = 0.0f;

    // Loop over tiles
    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        // Load tile from A into shared memory with boundary check
        int aCol = t * BLOCK_SIZE + threadIdx.x;
        if (row < N && aCol < N) {
            s_A[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory with boundary check
        int bRow = t * BLOCK_SIZE + threadIdx.y;
        if (bRow < N && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial product for this tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            value += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value to C with boundary check
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// C++ interface for PyTorch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Set up grid and block dimensions based on BLOCK_SIZE
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Matrix Multiplication with Block Size Tuning (CUDA)");
}
