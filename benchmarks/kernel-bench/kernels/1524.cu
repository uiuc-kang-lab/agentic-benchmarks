#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use a larger tile size to better amortize latency (adjustable if needed)
#define TILE_SIZE 32

// Optimized CUDA kernel for symmetric matrix multiplication
// This kernel reduces the number of __syncthreads() calls by preloading the first tile
// and then, inside the loop, synchronizing only before loading the next tile.
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float value = 0.0f;

    // Determine the number of tiles to iterate over
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Preload the first tile into shared memory
    
    int aCol = tile_idx * TILE_SIZE + tx;
    s_A[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;

    int bRow = tile_idx * TILE_SIZE + ty;
    s_B[ty][tx] = (col < N && bRow < N) ? B[bRow * N + col] : 0.0f;

    // Synchronize to ensure the first tile is fully loaded
    __syncthreads();

    // Loop over all tiles
    for (int t = 0; t < numTiles; t++) {
        // Multiply the elements of the current tile.
        // All threads use the shared memory tile that was preloaded.
        for (int k = 0; k < TILE_SIZE; k++) {
            value += s_A[ty][k] * s_B[k][tx];
        }
        // If not the last tile, load the next tile
        if (t < numTiles - 1) {
            // Synchronize to ensure all threads have finished using the current tile
            __syncthreads();
            int next = t + 1;
            int aColNext = next * TILE_SIZE + tx;
            s_A[ty][tx] = (row < N && aColNext < N) ? A[row * N + aColNext] : 0.0f;

            int bRowNext = next * TILE_SIZE + ty;
            s_B[ty][tx] = (col < N && bRowNext < N) ? B[bRowNext * N + col] : 0.0f;
        }
        // No extra synchronization is needed at the end of the last iteration
    }

    // Write the computed value back to global memory
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// C++ interface for the PyTorch extension
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

    // Determine grid and block dimensions
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Matrix Multiplication with Reduced __syncthreads() (CUDA)");
}
