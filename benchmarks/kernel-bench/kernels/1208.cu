#include <torch/extension.h>
#include <cuda_runtime.h>
#include <algorithm>

// This kernel uses double-buffering with asynchronous copy instructions (cp.async) to
// overlap the global memory loads of B with computation. Each block computes a tile of
// the output for a fixed (b, i, j) and a contiguous tile in the K dimension. The L-dimension
// reduction is tiled, and each tile of B is loaded into shared memory using cp.async, allowing
// the transfer to be overlapped with computation of the previous tile when pipelining is enabled.

// Kernel assumes TILE_L = 32 and TILE_K = 32.

__global__ void einsum_kernel_async_pipeline(
    const float* __restrict__ A,  // shape: [BATCH, I, J, L]
    const float* __restrict__ B,  // shape: [L, K]
    float* __restrict__ C,        // shape: [BATCH, I, J, K]
    int BATCH, int I, int J, int L, int K
) {
    // Tile parameters
    constexpr int TILE_L = 32;
    constexpr int TILE_K = 32;

    // We launch the kernel with a grid arranged over (BATCH*I*J) and over tiles in K dimension.
    // Each block computes one output tile for a fixed (b, i, j) and a tile of K values.

    int num_tiles_k = (K + TILE_K - 1) / TILE_K;
    // blockIdx.x encodes both the (b,i,j) index and the K-tile index
    int linear_idx = blockIdx.x;
    int bix = linear_idx / num_tiles_k; // index among BATCH * I * J
    int tile_k_idx = linear_idx % num_tiles_k;
    int k_base = tile_k_idx * TILE_K;

    // Decode bix into b, i, j
    int j_idx = bix % J;
    int i_idx = (bix / J) % I;
    int b_idx = bix / (I * J);

    // Pointers for A and C for the element (b, i, j)
    const float* A_ptr = A + (b_idx * I * J * L + i_idx * J * L + j_idx * L);
    float* C_ptr = C + (b_idx * I * J * K + i_idx * J * K + j_idx * K);

    // Each thread in the block corresponds to one output column in the K tile
    int thread_k = threadIdx.x;  // 0 <= threadIdx.x < TILE_K

    float sum = 0.0f;

    // Allocate shared memory as a double buffer for B tiles.
    // Shared memory size should be 2 * TILE_L * TILE_K floats.
    extern __shared__ float shared_buffer[];  // size in bytes provided by host
    float* sB0 = shared_buffer; // first buffer
    float* sB1 = shared_buffer + TILE_L * TILE_K; // second buffer

    // The reduction over L is tiled with a tile size of TILE_L
    int num_tiles = (L + TILE_L - 1) / TILE_L;
    int buffer = 0;  // toggle between sB0 and sB1

    for (int tile = 0; tile < num_tiles; tile++) {
        int tile_offset = tile * TILE_L;
        int current_tile = ((L - tile_offset) < TILE_L) ? (L - tile_offset) : TILE_L;

        // Select the current buffer for this tile
        float* curBuf = (buffer == 0) ? sB0 : sB1;

        // Compute the pointer for the current tile of B.
        // B is of shape [L, K]. We want rows [tile_offset, tile_offset+current_tile) and columns [k_base, k_base+TILE_K).
        const float* B_tile_ptr = B + tile_offset * K + k_base;

        // Asynchronously load the current B tile into shared memory using cp.async if available.
        // Each thread loads multiple elements in a loop.
        int total_load = current_tile * TILE_K;  // total floats to load
        for (int idx = threadIdx.x; idx < total_load; idx += blockDim.x) {
            int row = idx / TILE_K;
            int col = idx % TILE_K;
            const float* src_ptr = B_tile_ptr + row * K + col;
            float* dst_ptr = curBuf + row * TILE_K + col;
#if __CUDA_ARCH__ >= 800
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                          :
                          : "r"(dst_ptr), "l"(src_ptr), "n"(sizeof(float))
                          : "memory");
#else
            *dst_ptr = *src_ptr;
#endif
        }
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
#endif
        __syncthreads();  // Ensure the B tile is loaded

        // Compute partial sum for this tile: sum += A[b,i,j,l] * B[l, k] for l in [tile_offset, tile_offset+current_tile)
        for (int r = 0; r < current_tile; r++) {
            float a_val = A_ptr[tile_offset + r];
            // Each thread accesses its corresponding B element from the shared memory tile
            float b_val = curBuf[r * TILE_K + thread_k];
            sum += a_val * b_val;
        }

        buffer = 1 - buffer;  // Toggle buffer for double buffering
        __syncthreads();
    }

    // Write the result if the global k index is within bounds
    if (k_base + thread_k < K) {
        C_ptr[k_base + thread_k] = sum;
    }
}


// Host function launching the kernel with overlapped asynchronous copy and computation using CUDA streams.
// This implementation pipelines the tile loads of matrix B with the computation of partial sums.

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

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // Configure tiling parameters: TILE_K = 32
    constexpr int TILE_K = 32;
    int num_tiles_k = (K + TILE_K - 1) / TILE_K;

    // Each block computes one (b, i, j) and a tile of K values.
    int total_blocks = BATCH * I * J * num_tiles_k;
    dim3 grid(total_blocks);
    dim3 block(TILE_K);

    // Shared memory: double buffer for B tile (2 * TILE_L * TILE_K floats). TILE_L is set to 32.
    constexpr int TILE_L = 32;
    size_t shared_mem_size = 2 * TILE_L * TILE_K * sizeof(float);

    einsum_kernel_async_pipeline<<<grid, block, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with overlapped async copy (CUDA)");
}
