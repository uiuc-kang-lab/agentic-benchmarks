#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// This kernel flattens the usual 3D grid (batch, row, col) into a 2D grid by combining the batch dimension
// with the row tiling dimension. This can improve scheduling and allow more flexible mapping of threads
// to the problem domain. Each block computes one TILE_SIZE x TILE_SIZE tile of an output matrix from a single batch.
__global__ void flattened_bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Compute number of tile rows needed per batch
    int gridRows = (M + TILE_SIZE - 1) / TILE_SIZE;
    // Flattened grid: blockIdx.y carries both batch index and tile row index.
    int tile_row = blockIdx.y % gridRows;
    int batch = blockIdx.y / gridRows;
    int tile_col = blockIdx.x;  // blockIdx.x corresponds to tile column

    int row = tile_row * TILE_SIZE + threadIdx.y;
    int col = tile_col * TILE_SIZE + threadIdx.x;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Pointers to the current batch's matrix A and B
    const float* a_batch = A + batch * M * K;
    const float* b_batch = B + batch * K * N;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int A_col = t * TILE_SIZE + threadIdx.x;
        int B_row = t * TILE_SIZE + threadIdx.y;

        // Load tile from A into shared memory with boundary check
        if (row < M && A_col < K) {
            As[threadIdx.y][threadIdx.x] = a_batch[row * K + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory with boundary check
        if (B_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = b_batch[B_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();
        if (t + 1 < numTiles) {
            // Preload next tiles into shared memory
            int next_A_col = (t + 1) * TILE_SIZE + threadIdx.x;
            int next_B_row = (t + 1) * TILE_SIZE + threadIdx.y;
            if (row < M && next_A_col < K) {
                As[threadIdx.y][threadIdx.x] = a_batch[row * K + next_A_col];
            }
            if (next_B_row < K && col < N) {
                Bs[threadIdx.y][threadIdx.x] = b_batch[next_B_row * N + col];
            }
        }

        // Compute partial products for this tile
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }

        __syncthreads();
        if (t + 1 < numTiles) {
            // Preload next tiles into shared memory
            int next_A_col = (t + 1) * TILE_SIZE + threadIdx.x;
            int next_B_row = (t + 1) * TILE_SIZE + threadIdx.y;
            if (row < M && next_A_col < K) {
                As[threadIdx.y][threadIdx.x] = a_batch[row * K + next_A_col];
            }
            if (next_B_row < K && col < N) {
                Bs[threadIdx.y][threadIdx.x] = b_batch[next_B_row * N + col];
            }
        }
    }

    // Write the computed value to C if within valid range
    if (row < M && col < N) {
        C[batch * M * N + row * N + col] = sum;
    }
}

// Forward function to launch the kernel
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

    int gridCols = (N + TILE_SIZE - 1) / TILE_SIZE;
    int gridRows = (M + TILE_SIZE - 1) / TILE_SIZE;
    // Flatten the grid: gridDim.x = gridCols, gridDim.y = batch_size * gridRows
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(gridCols, batch_size * gridRows);

    flattened_bmm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with flattened grid indexing (CUDA)");
}
