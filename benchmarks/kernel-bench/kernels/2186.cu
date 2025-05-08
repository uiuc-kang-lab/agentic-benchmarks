#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

// Define tile sizes
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

// Kernel using shared memory to improve data reuse and reduce global memory accesses
__global__ void tiledSharedMemoryKernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int K,
                                        int M,
                                        int N) {
    // Shared memory allocation for tiles
    __shared__ float As[TILE_K][TILE_M];
    __shared__ float Bs[TILE_K][TILE_N];

    // Compute global indices
    int row = blockIdx.x * TILE_M + threadIdx.x;
    int col = blockIdx.y * TILE_N + threadIdx.y;
    float sum = 0.0f;

    // Loop over tiles of K dimension
    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        // Load data into shared memory
        if (t * TILE_K + threadIdx.y < K && row < M)
            As[threadIdx.y][threadIdx.x] = A[(t * TILE_K + threadIdx.y) * M + row];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (t * TILE_K + threadIdx.x < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_K + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product for this tile
        for (int k = 0; k < TILE_K; ++k) {
            sum += As[k][threadIdx.x] * Bs[k][threadIdx.y];
        }

        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// The forward function exposed via PyBind11
// Inputs:
//   A: Tensor of shape (K, M) [CUDA, float32]
//   B: Tensor of shape (K, N) [CUDA, float32]
// Returns:
//   C: Tensor of shape (M, N) computed as A.T * B.
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors and of type float32.
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // A: (K, M) and B: (K, N)
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C of shape (M, N).
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define thread block and grid sizes.
    dim3 blockDim(TILE_M, TILE_N);
    dim3 gridDim((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    // Get raw pointers to the data.
    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the CUDA kernel.
    tiledSharedMemoryKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using tiled shared memory (CUDA)");
}
