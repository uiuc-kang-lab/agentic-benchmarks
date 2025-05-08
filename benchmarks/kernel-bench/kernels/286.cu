#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size for shared memory tiling
#define TILE 16
#define VECTOR_SIZE 4  // Unroll factor (assumes TILE is divisible by VECTOR_SIZE)

// Device function to load one element from global memory into a shared memory tile
// using thread indices for coordinated loading with boundary checks.
__device__ __forceinline__ void load_tile(const float* __restrict__ matrix, float tile[TILE][TILE], int row, int col, int dim0, int dim1) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    if (row < dim0 && col < dim1) {
        tile[ty][tx] = matrix[row * dim1 + col];
    } else {
        tile[ty][tx] = 0.0f;
    }
}

// Combined kernel: uses modular shared memory loads (as in kernel 2) and manual inner-loop unrolling (as in kernel 1).
// It also leverages extern shared memory to dynamically allocate two tiles (for A and B).
__global__ void bmm_tiled_unrolled_modular_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int batch_size,
    const int M,
    const int K,
    const int N
) {
    // Allocate shared memory for tiles of A and B
    extern __shared__ float shared[];
    float (*As)[TILE] = (float (*)[TILE])shared;
    float (*Bs)[TILE] = (float (*)[TILE])(shared + TILE * TILE);

    // Block indices: bx, by determine the tile, bz determines the batch index
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute row and column for the output matrix
    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;

    // Pre-calculate batch offsets
    const int a_batch_offset = bz * M * K;
    const int b_batch_offset = bz * K * N;

    // Loop over tiles along the K dimension
    int numTiles = (K + TILE - 1) / TILE;
    for (int t = 0; t < numTiles; t++) {
        // Coordinates within A and B for the current tile
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;

        // Load tile from A and B into shared memory using the modular load function
        load_tile(A + a_batch_offset, As, row, a_col, M, K);
        load_tile(B + b_batch_offset, Bs, b_row, col, K, N);

        __syncthreads();

        // Compute partial product for the current tile with manual unrolling of the inner loop
        #pragma unroll
        for (int k = 0; k < TILE; k += VECTOR_SIZE) {
            sum += As[ty][k]     * Bs[k][tx];
            sum += As[ty][k + 1] * Bs[k + 1][tx];
            sum += As[ty][k + 2] * Bs[k + 2][tx];
            sum += As[ty][k + 3] * Bs[k + 3][tx];
        }

        __syncthreads();
    }

    // Write the computed value to output matrix if within valid bounds
    if (row < M && col < N) {
        C[bz * M * N + row * N + col] = sum;
    }
}

// Host interface for PyTorch extension
torch::Tensor forward_bmm_tiled_unrolled_modular(torch::Tensor A, torch::Tensor B) {
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

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch_size);

    size_t shared_mem_size = 2 * TILE * TILE * sizeof(float);

    bmm_tiled_unrolled_modular_kernel<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm_tiled_unrolled_modular, "Batched matrix multiplication with tiled shared memory, modular loads, and manual unrolling (CUDA)");
}
