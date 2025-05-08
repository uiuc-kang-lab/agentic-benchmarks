#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Tile dimensions: using a smaller output tile to reduce register pressure
// while unifying shared memory loading for better memory coalescing.
#define TILE_SIZE 16
#define TILE_K 32

// Macros to check inputs
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel computes matrix multiplication C = A x B.
// Each CUDA block computes a TILE_SIZE x TILE_SIZE tile of C.
// The K dimension is processed in chunks of TILE_K.
// A unified shared memory loading scheme is used: all threads 
// cooperatively load the required subtiles of A and B from global memory
// into shared memory in a strided manner. Then each thread computes its
// output element as a dot product over the TILE_K (or remaining) elements.

__global__ void tiled_unified_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       const int M, const int N, const int K) {
    // Compute global row and col indices for C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;  // Accumulator for the dot product

    // Number of tiles to iterate over in the K dimension
    int numTiles = (K + TILE_K - 1) / TILE_K;

    // Shared memory for the current tile of A and B
    __shared__ float sA[TILE_SIZE][TILE_K];  // Tile of A: dimensions TILE_SIZE x TILE_K
    __shared__ float sB[TILE_K][TILE_SIZE];    // Tile of B: dimensions TILE_K x TILE_SIZE

    // Unified loading: We use all threads in the block to load both sA and sB.
    // Total number of elements to load in shared memory for one tile phase:
    // For sA: TILE_SIZE * TILE_K, for sB: TILE_K * TILE_SIZE
    // Total load = 2 * TILE_SIZE * TILE_K
    int total_load = 2 * TILE_SIZE * TILE_K;
    int tid = threadIdx.y * TILE_SIZE + threadIdx.x;  // flatten block thread index

    // Loop over tiles in the K dimension
    for (int t = 0; t < numTiles; t++) {
        // Cooperative loading into shared memory in a strided loop
        for (int i = tid; i < total_load; i += TILE_SIZE * TILE_SIZE) {
            if (i < TILE_SIZE * TILE_K) {
                // Loading for sA
                int a_row = i / TILE_K;         // local row in sA
                int a_col = i % TILE_K;           // local col in sA
                int global_a_row = blockIdx.y * TILE_SIZE + a_row;
                int global_a_col = t * TILE_K + a_col;
                if (global_a_row < M && global_a_col < K)
                    sA[a_row][a_col] = A[global_a_row * K + global_a_col];
                else
                    sA[a_row][a_col] = 0.0f;
            } else {
                // Loading for sB
                int idx = i - TILE_SIZE * TILE_K;
                int b_row = idx / TILE_SIZE;    // local row in sB
                int b_col = idx % TILE_SIZE;    // local col in sB
                int global_b_row = t * TILE_K + b_row;
                int global_b_col = blockIdx.x * TILE_SIZE + b_col;
                if (global_b_row < K && global_b_col < N)
                    sB[b_row][b_col] = B[global_b_row * N + global_b_col];
                else
                    sB[b_row][b_col] = 0.0f;
            }
        }
        __syncthreads(); // Ensure the shared memory tiles are fully loaded

        // Determine effective TILE_K for the last tile if K is not a multiple of TILE_K
        int effectiveTILEK = TILE_K;
        if (t == numTiles - 1 && (K % TILE_K) != 0)
            effectiveTILEK = K % TILE_K;

        // Compute the dot product for the tile
        #pragma unroll
        for (int k = 0; k < effectiveTILEK; k++) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads(); // Synchronize before loading the next tile
    }

    // Write the computed value to C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Host function to launch the kernel
void matrix_multiply_cuda(const torch::Tensor &A,
                            const torch::Tensor &B,
                            torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // Configure kernel launch dimensions based on a 2D tiling of C
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                    (M + TILE_SIZE - 1) / TILE_SIZE);

    tiled_unified_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}

// PyTorch binding
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int N = B.size(1);

    auto options = torch::TensorOptions()
        .dtype(A.dtype())
        .device(A.device())
        .requires_grad(false);
    
    torch::Tensor C = torch::empty({M, N}, options);
    matrix_multiply_cuda(A, B, C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with unified shared memory loading (CUDA)");
}
