#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define warp and tile dimensions
#define WARP_SIZE 32
#define TILE_M 8    // Number of output rows computed per block
#define TILE_N 8    // Number of output columns computed per block

// CUDA kernel: Computes C = A.T * B using warp-level parallelism
// A: shape (K, M) stored in row-major order (A[k][i] -> A[k*M + i])
// B: shape (K, N) stored in row-major order (B[k][j] -> B[k*N + j])
// C: shape (M, N) stored in row-major order (C[i][j] -> C[i*N + j])
__global__ void warpDotKernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int K, int M, int N) {
    // Block handles TILE_M rows and TILE_N columns of output
    // Each warp processes one element in the output tile
    const unsigned int warp_size = WARP_SIZE;
    const unsigned int lane = threadIdx.x % warp_size;
    const unsigned int warp_id = threadIdx.x / warp_size;
    
    // Calculate output position
    const unsigned int row = blockIdx.y * TILE_M + (warp_id / TILE_N);
    const unsigned int col = blockIdx.x * TILE_N + (warp_id % TILE_N);
    
    // Initialize accumulator
    float sum = 0.0f;
    
    // Only compute if within bounds
    if (row < M && col < N) {
        // Each thread in warp handles part of reduction
        #pragma unroll 4
        for (int k = lane; k < K; k += warp_size) {
            const float a_val = __ldg(&A[k * M + row]);  // Load from A.T position
            const float b_val = __ldg(&B[k * N + col]);
            sum += a_val * b_val;
        }
        
        // Warp-level reduction using shuffle
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xffffffff, sum, offset);
        }
        
        // First thread in warp writes result
        if (lane == 0) {
            C[row * N + col] = sum;
        }
    }
}

// The forward function exposed via PyBind11.
// Inputs:
//   A: Tensor of shape (K, M) (CUDA, float32)
//   B: Tensor of shape (K, N) (CUDA, float32)
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B.
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N)
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define grid and block dimensions.
    // Each block computes a tile of size (TILE_M x TILE_N) elements of C.
    // Each element of C is computed by one warp of WARP_SIZE threads.
    dim3 blockDim(TILE_N * WARP_SIZE, TILE_M);
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    warpDotKernel<<<grid, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using warp-level primitives (CUDA)");
}
