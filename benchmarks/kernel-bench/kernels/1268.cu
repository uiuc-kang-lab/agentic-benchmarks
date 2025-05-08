#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile dimensions for the L and K dimensions
#define TILE_L 32
#define TILE_K 32

// This kernel computes C[b,i,j,k] = sum_{l=0}^{L-1} A[b,i,j,l] * B[l,k] for a 4D tensor A and 2D matrix B.
// Each block is assigned to one (batch, i, j) group and a tile in the k dimension. Threads in the block compute a contiguous
// set of k outputs. To improve performance, we use shared memory with double buffering to load tiles of B from global
// memory. Double buffering allows overlapping the preload of the next tile with the computation of the current tile,
// thus reducing the number of required __syncthreads() calls. Only one synchronization per tile iteration is used for
// shared memory consistency.

__global__ void einsum_kernel_double_buffer(
    const float* __restrict__ A,       // A is of shape [BATCH * I * J, L]
    const float* __restrict__ B,       // B is of shape [L, K]
    float* __restrict__ C,             // C is of shape [BATCH * I * J, K]
    int IJP,                           // Total number of groups = BATCH * I * J
    int L,
    int K
) {
    // Each block handles a specific (batch, i, j) group (indexed by p) and a tile of the K dimension.
    int p = blockIdx.x; 
    int k_start = blockIdx.y * TILE_K;
    int tid = threadIdx.x;  // Expect blockDim.x == TILE_K
    int k = k_start + tid;

    // Offsets into A and C for the current (b,i,j) group
    int offsetA = p * L;
    int offsetC = p * K;

    float sum = 0.f;

    // Allocate double buffers in shared memory for a tile of B.
    // Each buffer holds TILE_L rows and TILE_K columns of B (stored in row-major order).
    __shared__ float shared_B[2][TILE_L * TILE_K];

    // Number of tiles to cover the L dimension
    int numTiles = (L + TILE_L - 1) / TILE_L;

    // 'curr' selects the buffer to use for computation
    int curr = 0;

    // Preload the first tile of B into shared_B[curr]
    int l0 = 0; // tile start index
    #pragma unroll
        for (int i = 0; i < TILE_L; i++) {
        int l_idx = l0 + i;
        float b_val = (l_idx < L && k < K) ? B[l_idx * K + k] : 0.f;
        shared_B[curr][i * TILE_K + tid] = b_val;
    }
    __syncthreads(); // Ensure the first tile is fully loaded before computation

    // Loop over tiles in the L dimension
    for (int t = 0; t < numTiles; t++) {
        int l_tile = t * TILE_L;

        // If not processing the last tile, preload the next tile into the alternate buffer
        if (t < numTiles - 1) {
            int next = 1 - curr;
            int l_next = (t + 1) * TILE_L;
            #pragma unroll
        for (int i = 0; i < TILE_L; i++) {
                int l_idx = l_next + i;
                float b_val = (l_idx < L && k < K) ? B[l_idx * K + k] : 0.f;
                shared_B[next][i * TILE_K + tid] = b_val;
            }
        }

        __syncthreads(); // Synchronize to ensure that the preload (if any) is complete

        // Compute partial dot-product for the current tile from shared memory
        #pragma unroll
        for (int i = 0; i < TILE_L; i++) {
            int l_idx = l_tile + i;
            if (l_idx < L) {
                float a_val = A[offsetA + l_idx];
                sum += a_val * shared_B[curr][i * TILE_K + tid];
            }
        }

        // Swap buffers if a next tile was preloaded
        if (t < numTiles - 1) {
            curr = 1 - curr;
        }
        // No additional __syncthreads() after computation is needed because the __syncthreads() at the start
        // of the next iteration will ensure proper ordering across threads.
    }

    if (k < K) {
        C[offsetC + k] = sum;
    }
}

// The forward function validates inputs, sets up grid dimensions, and launches the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 4, "A must be 4D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "Dimension mismatch in L");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);
    
    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // Flatten (BATCH, I, J) into a single dimension
    int IJP = BATCH * I * J;
    
    // Grid: one block per (batch,i,j) group and a tile in the k dimension
    dim3 blocks(IJP, (K + TILE_K - 1) / TILE_K);
    dim3 threads(TILE_K);

    einsum_kernel_double_buffer<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        IJP, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with double buffering (CUDA)");
}
