#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel computes C = A^T * B^T, where A has shape (K, M) and B has shape (N, K).
// Equivalently, defining A' = A^T (shape: M x K) and B' = B^T (shape: K x N), we have C = A' * B'.
// We use tiling with shared memory and refactor the boundary condition checks to minimize divergence.

template <typename scalar_t>
__global__ void matmul_transpose_nodiv_shared_kernel(
    const scalar_t* __restrict__ A, // A: (K, M), accessed as A[k * M + row] for A' (M x K)
    const scalar_t* __restrict__ B, // B: (N, K), accessed as B[col * K + k] for B' (K x N)
    scalar_t* __restrict__ C,
    const int M, // number of rows in C and A'
    const int N, // number of columns in C and B'
    const int K  // reduction dimension
) {
    const int TILE_SIZE = 32;
    // Using standard GEMM tiling on transposed matrices: we compute C[r, c] = sum_{k} A'[r, k] * B'[k, c]
    // where A'[r, k] = A[k * M + r] and B'[k, c] = B[c * K + k].
    
    // Map block indices to output matrix coordinates
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // corresponds to r in A' and C
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // corresponds to c in B' and C
    
    scalar_t sum = 0;

    // Shared memory tiles with padding to avoid bank conflicts
    __shared__ scalar_t As[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t Bs[TILE_SIZE][TILE_SIZE];

    // Number of tiles to iterate over in the k dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int tiledK = t * TILE_SIZE;
        // Check if the tile is completely within bounds
        bool fullTile = (tiledK + TILE_SIZE <= K);

        // Load the tile from A' into shared memory.
        // A' has dimensions (M, K) and is stored in A as A[k*M + row].
        if (fullTile) {
            // Uniform load without per-thread condition, since all indices in the tile are valid.
            As[threadIdx.y][threadIdx.x] = (row < M) ? A[(tiledK + threadIdx.x) * M + row] : scalar_t(0);
        } else {
            int kA = tiledK + threadIdx.x; // k index for A'
            As[threadIdx.y][threadIdx.x] = ((kA < K) && (row < M)) ? A[kA * M + row] : scalar_t(0);
        }

        // Load the tile from B' into shared memory.
        // B' has dimensions (K, N) and is stored in B as B[col*K + k].
        if (fullTile) {
            Bs[threadIdx.y][threadIdx.x] = (col < N) ? B[col * K + (tiledK + threadIdx.y)] : scalar_t(0);
        } else {
            int kB = tiledK + threadIdx.y; // k index for B'
            Bs[threadIdx.y][threadIdx.x] = ((kB < K) && (col < N)) ? B[col * K + kB] : scalar_t(0);
        }

        __syncthreads();

        // Multiply the two tiles together; inner loop contains no divergent conditionals
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result to global memory if within valid bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch the kernel
// A: shape (K, M), B: shape (N, K), C: shape (M, N)
// This kernel computes C = A^T * B^T by first interpreting A^T and B^T as A' and B'.

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());
    
    const int TILE_SIZE = 32;
    // Grid dimensions: blocks.x covers columns, blocks.y covers rows
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_nodiv_shared_kernel", ([&] {
        matmul_transpose_nodiv_shared_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed matrices forward (CUDA) - minimized warp divergence");
}
