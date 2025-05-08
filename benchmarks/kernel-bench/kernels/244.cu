#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Batched matrix multiplication using double buffering with minimal synchronization
// A: (batch_size, M, K), B: (batch_size, K, N), C: (batch_size, M, N)
__global__ void bmm_double_buffer_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Determine batch, row, and column indices
    int b = blockIdx.z;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Double buffers in shared memory for tiles of A and B
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    // Base pointers for the current batch
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int current = 0;
    int next = 1;

    // Preload the first tile into the current buffer
    {
        int tile = 0;
        int aCol = tile * TILE_SIZE + threadIdx.x;
        int bRow = tile * TILE_SIZE + threadIdx.y;

        if (row < M && aCol < K)
            As[current][threadIdx.y][threadIdx.x] = batch_A[row * K + aCol];
        else
            As[current][threadIdx.y][threadIdx.x] = 0.0f;

        if (bRow < K && col < N)
            Bs[current][threadIdx.y][threadIdx.x] = batch_B[bRow * N + col];
        else
            Bs[current][threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads(); // Ensure the first tile is loaded

    // Loop over all tiles
    for (int tile = 0; tile < numTiles; tile++) {
        // If there is a next tile, preload it into the next buffer
        if (tile < numTiles - 1) {
            int nextTile = tile + 1;
            int aColNext = nextTile * TILE_SIZE + threadIdx.x;
            int bRowNext = nextTile * TILE_SIZE + threadIdx.y;
            
            if (row < M && aColNext < K)
                As[next][threadIdx.y][threadIdx.x] = batch_A[row * K + aColNext];
            else
                As[next][threadIdx.y][threadIdx.x] = 0.0f;
            
            if (bRowNext < K && col < N)
                Bs[next][threadIdx.y][threadIdx.x] = batch_B[bRowNext * N + col];
            else
                Bs[next][threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Compute partial dot product using the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[current][threadIdx.y][k] * Bs[current][k][threadIdx.x];
        }
        
        // If a next tile was loaded, synchronize once and swap buffers
        if (tile < numTiles - 1) {
            __syncthreads(); // Ensure the next tile is fully loaded
            // Swap the current and next buffer indices
            current ^= 1;
            next ^= 1;
        }
    }

    // Write the result to global memory if within valid output bounds
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
    }
}

// Host function to launch the kernel
torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE,
              batch_size);

    bmm_double_buffer_kernel<<<grid, block>>>(
         A.data_ptr<float>(),
         B.data_ptr<float>(),
         C.data_ptr<float>(),
         batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with double buffering and minimal synchronization (CUDA)");
}
