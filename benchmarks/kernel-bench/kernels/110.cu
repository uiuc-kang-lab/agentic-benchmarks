#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Tile dimensions
#define TILE_M 4
#define TILE_N 4
#define TILE_K 32

// Macros for checking inputs
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel: Each block computes a TILE_M x TILE_N tile of C.
// Within each block, each warp (32 threads) computes one output element using
// shared memory for the A and B tiles and warp-level reduction via __shfl_down_sync().
__global__ void sharedWarpReductionMatMulKernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                int M, int N, int K) {
    // Determine the block's base position in C
    int block_row = blockIdx.y;
    int block_col = blockIdx.x;
    int base_row = block_row * TILE_M;
    int base_col = block_col * TILE_N;

    // Each block is organized into TILE_M*TILE_N warps.
    // Each warp (32 threads) computes one output element in the tile.
    int warp_id = threadIdx.x / 32;      // Warp index in the block (0 .. TILE_M*TILE_N - 1)
    int lane = threadIdx.x % 32;           // Lane index within the warp

    int local_row = warp_id / TILE_N;      // Row index within the tile [0, TILE_M)
    int local_col = warp_id % TILE_N;      // Column index within the tile [0, TILE_N)

    int global_row = base_row + local_row; // Global row index for the C element
    int global_col = base_col + local_col; // Global col index for the C element

    float acc = 0.0f;

    // Shared memory for the current tile of A and B
    __shared__ float sA[TILE_M][TILE_K];  // A tile: dimensions TILE_M x TILE_K
    // Note: For B, we store a tile with dimensions TILE_K x TILE_N
    __shared__ float sB[TILE_K][TILE_N];

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_K - 1) / TILE_K;
    
    for (int t = 0; t < numTiles; t++) {
        // Total elements to load: A tile has TILE_M*TILE_K, B tile has TILE_K*TILE_N
        const int totalA = TILE_M * TILE_K;
        const int totalB = TILE_K * TILE_N;
        const int totalLoad = totalA + totalB;  // = 128 + 128 = 256 when TILE_M=TILE_N=4 and TILE_K=32

        // Each thread loads elements into shared memory in a strided manner
        for (int i = threadIdx.x; i < totalLoad; i += blockDim.x) {
            if (i < totalA) {
                int a_row = i / TILE_K;       // Row in A tile [0, TILE_M)
                int a_col = i % TILE_K;         // Col in A tile [0, TILE_K)
                int A_row = block_row * TILE_M + a_row;
                int A_col = t * TILE_K + a_col;
                if (A_row < M && A_col < K)
                    sA[a_row][a_col] = A[A_row * K + A_col];
                else
                    sA[a_row][a_col] = 0.0f;
            } else {
                int j = i - totalA;
                int b_row = j / TILE_N;       // Row in B tile, corresponds to k index
                int b_col = j % TILE_N;       // Col in B tile [0, TILE_N)
                int B_row = t * TILE_K + b_row;
                int B_col = block_col * TILE_N + b_col;
                if (B_row < K && B_col < N)
                    sB[b_row][b_col] = B[B_row * N + B_col];
                else
                    sB[b_row][b_col] = 0.0f;
            }
        }
        __syncthreads(); // Ensure the tile is loaded

        // Determine the effective tile width (for boundary cases)
        int effectiveTILE_K = TILE_K;
        if (t == numTiles - 1 && (K % TILE_K) != 0) {
            effectiveTILE_K = K % TILE_K;
        }

        // Each warp computes a partial dot product for its assigned output element.
        // The dot product is computed in a strided manner over the tile's K dimension.
        float sum = 0.0f;
        for (int k = lane; k < effectiveTILE_K; k += 32) {
            float a_val = sA[local_row][k];
            float b_val = sB[k][local_col];
            sum += a_val * b_val;
        }

        // Intra-warp reduction using warp shuffle
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        // The first lane of each warp adds the reduced sum to the accumulator
        if (lane == 0) {
            acc += sum;
        }
        __syncthreads(); // Prepare for loading the next tile
    }

    // Write the computed result to C (only one thread per warp writes the outcome)
    if (global_row < M && global_col < N) {
        if (lane == 0) {
            C[global_row * N + global_col] = acc;
        }
    }
}

// Host function that sets up kernel launch parameters
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    // Grid dimensions based on output matrix C tiling
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    // Each block has (TILE_M * TILE_N * 32) threads; here 4*4*32 = 512 threads
    int blockThreads = TILE_M * TILE_N * 32;

    sharedWarpReductionMatMulKernel<<<grid, blockThreads>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication using shared memory, intra-block reduction and warp-level primitives (CUDA)");
}
