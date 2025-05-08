#include <torch/extension.h>
#include <cuda_runtime.h>

// Define tiling and block dimensions
#define TILE_SIZE 32
#define BLOCK_ROW 8
#define BLOCK_COL 32

// Inline function to perform asynchronous copy of one float from global to shared memory using cp.async
__device__ inline void cp_async_f32(float* dst, const float* src) {
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" 
                  : 
                  : "r"(dst), "l"(src), "n"(4));
#else
    *dst = *src;
#endif
}

// Kernel: reinterpret the 4D tensor A (reshaped as [R, L] where R = BATCH*I*J) and B ([L, K]) to perform GEMV
// using double-buffered asynchronous copies (cp.async) of tiles of B into shared memory.
// Each thread computes one element of C (of shape [R, K]) corresponding to one dot-product.

__global__ void tiled_cp_async_kernel(
    const float* __restrict__ A, // [R, L] where R = BATCH * I * J
    const float* __restrict__ B, // [L, K]
    float* __restrict__ C,       // [R, K]
    int R, int L, int K
) {
    // Compute output indices for C
    int c = blockIdx.x * BLOCK_COL + threadIdx.x;  // column index in [0, K)
    int r = blockIdx.y * BLOCK_ROW + threadIdx.y;    // row index in [0, R)
    if (r >= R || c >= K) return;

    float sum = 0.0f;

    // Allocate shared memory for double buffering of B tiles
    // Each buffer holds a tile of B: TILE_SIZE rows and BLOCK_COL columns
    __shared__ float sB[2][TILE_SIZE][BLOCK_COL];

    // Number of tiles needed to cover the L dimension
    int ntiles = (L + TILE_SIZE - 1) / TILE_SIZE;
    int buffer_index = 0;  // current buffer index

    // PREFETCH: Load the first tile (tile 0) of B into sB[buffer_index] asynchronously
    int tile0_offset = 0;
    if (tile0_offset < L) {
        // Each thread in the block cooperatively loads elements of the tile
        // Total elements in the tile = TILE_SIZE * BLOCK_COL
        for (int t = threadIdx.x; t < TILE_SIZE * BLOCK_COL; t += BLOCK_COL) {
            int local_t = t / BLOCK_COL;   // row in the tile
            int local_c = t % BLOCK_COL;   // column in the tile
            int global_c = blockIdx.x * BLOCK_COL + local_c; // corresponding global column index in B
            if (global_c < K && (tile0_offset + local_t) < L) {
                const float* src = B + (tile0_offset + local_t) * K + global_c;
                float* dst = &sB[buffer_index][local_t][local_c];
                cp_async_f32(dst, src);
            }
        }
    }
    __syncthreads(); // Ensure the first tile is loaded before computation

    // Pipeline through tiles with double buffering
    // Loop over tiles 1 to ntiles-1: compute on the previously prefetched tile while prefetching the next tile
    for (int tile = 1; tile < ntiles; tile++) {
        int current_tile_offset = (tile - 1) * TILE_SIZE;
        // Compute partial dot-product using the tile that was prefetched in sB[buffer_index]
        for (int t = 0; t < TILE_SIZE; t++) {
            int l_idx = current_tile_offset + t;
            if (l_idx < L) {
                // Each thread loads its corresponding element from A
                float a_val = A[r * L + l_idx];
                // sB buffer holds the B tile for the current block's columns
                float b_val = sB[buffer_index][t][threadIdx.x];
                sum += a_val * b_val;
            }
        }

        // Prefetch the next tile into the alternate buffer
        int next_buffer = 1 - buffer_index;
        int next_tile_offset = tile * TILE_SIZE;
        for (int t = threadIdx.x; t < TILE_SIZE * BLOCK_COL; t += BLOCK_COL) {
            int local_t = t / BLOCK_COL;
            int local_c = t % BLOCK_COL;
            int global_c = blockIdx.x * BLOCK_COL + local_c;
            if (global_c < K && (next_tile_offset + local_t) < L) {
                const float* src = B + (next_tile_offset + local_t) * K + global_c;
                float* dst = &sB[next_buffer][local_t][local_c];
                cp_async_f32(dst, src);
            }
        }
        __syncthreads(); // Wait for the asynchronous copy of the next tile to complete
        buffer_index = next_buffer; // Swap buffers
    }

    // Process the last prefetched tile
    int last_tile_offset = (ntiles - 1) * TILE_SIZE;
    for (int t = 0; t < TILE_SIZE; t++) {
        int l_idx = last_tile_offset + t;
        if (l_idx < L) {
            float a_val = A[r * L + l_idx];
            float b_val = sB[buffer_index][t][threadIdx.x];
            sum += a_val * b_val;
        }
    }

    // Write the computed dot product into the output matrix C
    C[r * K + c] = sum;
}

// Host function: reinterpret A as [R, L] and output C as [R, K], then reshape back to [BATCH, I, J, K].
// A is originally a 4D tensor of shape [BATCH, I, J, L] and B is [L, K].

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in l");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);
    int R = BATCH * I * J;  // Fuse the first three dimensions of A

    // Create output tensor C of shape [R, K]
    auto C = torch::zeros({R, K}, A.options());

    // Configure 2D grid and block dimensions
    dim3 block(BLOCK_COL, BLOCK_ROW);
    dim3 grid((K + BLOCK_COL - 1) / BLOCK_COL, (R + BLOCK_ROW - 1) / BLOCK_ROW);

    tiled_cp_async_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        R, L, K
    );

    // Reshape the output back to [BATCH, I, J, K]
    return C.view({BATCH, I, J, K});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled GEMV with asynchronous cp.async for overlapping computation and memory transfers (CUDA)");
}
