#include <torch/extension.h>
#include <cuda_runtime.h>

// Define tile dimensions and register tile factors
#define TILE_DIM 32
#define THREAD_TILE_M 2
#define THREAD_TILE_N 2

// Kernel implementing register tiling for matrix multiplication with balanced workload distribution
__global__ void register_tile_matmul_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int M, int K, int N) {
    // Compute block starting indices in output C
    int blockRow = blockIdx.y * TILE_DIM;
    int blockCol = blockIdx.x * TILE_DIM;

    // Each block has (TILE_DIM/THREAD_TILE_N) x (TILE_DIM/THREAD_TILE_M) threads
    int tx = threadIdx.x; // ranges 0 to (TILE_DIM/THREAD_TILE_N - 1)
    int ty = threadIdx.y; // ranges 0 to (TILE_DIM/THREAD_TILE_M - 1)

    // Each thread computes a sub-tile of size THREAD_TILE_M x THREAD_TILE_N
    int row_start = blockRow + ty * THREAD_TILE_M;
    int col_start = blockCol + tx * THREAD_TILE_N;

    // Initialize register accumulators for the sub-tile
    float Creg[THREAD_TILE_M][THREAD_TILE_N];
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_N; ++j) {
            Creg[i][j] = 0.0f;
        }
    }

    // Shared memory tiles for A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Number of tiles over the K dimension
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    // Loop over tiles in the K dimension
    for (int t = 0; t < numTiles; t++) {
        // Total number of elements per tile
        const int numElements = TILE_DIM * TILE_DIM; // 1024 elements
        // Total threads per block
        const int threadsPerBlock = blockDim.x * blockDim.y; // expected to be 256
        // Each thread loads multiple elements evenly
        int threadId = threadIdx.y * blockDim.x + threadIdx.x;
        int elementsPerThread = (numElements + threadsPerBlock - 1) / threadsPerBlock; // should be 4

        // Load data into shared memory tile for A
        for (int i = 0; i < elementsPerThread; i++) {
            int index = threadId + i * threadsPerBlock;
            if (index < numElements) {
                int r = index / TILE_DIM;
                int c = index % TILE_DIM;
                int Arow = blockRow + r;
                int Acol = t * TILE_DIM + c;
                As[r][c] = (Arow < M && Acol < K) ? A[Arow * K + Acol] : 0.0f;
            }
        }

        // Load data into shared memory tile for B
        for (int i = 0; i < elementsPerThread; i++) {
            int index = threadId + i * threadsPerBlock;
            if (index < numElements) {
                int r = index / TILE_DIM;
                int c = index % TILE_DIM;
                int Brow = t * TILE_DIM + r;
                int Bcol = blockCol + c;
                Bs[r][c] = (Brow < K && Bcol < N) ? B[Brow * N + Bcol] : 0.0f;
            }
        }
        __syncthreads();

        // Compute partial product for the current tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            // Each thread loads values from the shared memory tiles and accumulates
            #pragma unroll
            for (int i = 0; i < THREAD_TILE_M; i++) {
                float a_val = As[ty * THREAD_TILE_M + i][k];
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_N; j++) {
                    Creg[i][j] += a_val * Bs[k][tx * THREAD_TILE_N + j];
                }
            }
        }
        __syncthreads();
    }

    // Write the computed sub-tile results back to global memory
    #pragma unroll
    for (int i = 0; i < THREAD_TILE_M; i++) {
        for (int j = 0; j < THREAD_TILE_N; j++) {
            int Crow = row_start + i;
            int Ccol = col_start + j;
            if (Crow < M && Ccol < N) {
                C[Crow * N + Ccol] = Creg[i][j];
            }
        }
    }
}

// The forward function wraps the kernel launch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "Tensor B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Define block and grid dimensions for balanced workload distribution
    dim3 blockDim(TILE_DIM / THREAD_TILE_N, TILE_DIM / THREAD_TILE_M); // (32/2, 32/2) = (16,16) => 256 threads per block
    dim3 gridDim((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    register_tile_matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Register-tiled matrix multiplication with even workload distribution (CUDA)");
}
