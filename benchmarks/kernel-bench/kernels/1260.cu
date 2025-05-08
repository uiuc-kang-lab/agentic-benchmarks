#include <torch/extension.h>
#include <cuda_runtime.h>

// Tile size for the K dimension
#define TILE_K 32

// This kernel computes: C[b,i,j,k] = sum_{l} A[b,i,j,l] * B[l,k] using shared memory for B to reduce global memory accesses.
// It maps each block to a unique (b, i, j) group and a tile along the k dimension.
// The inner loop over l is unrolled by a factor of 4 to reduce loop overhead and improve ILP.
// Memory accesses for B and C are coalesced, while A is broadcast to all threads in the block.
__global__ void einsum_kernel_combined(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int IJP,   // total number of (batch * I * J) groups
    int L,
    int K
) {
    // Each block is responsible for one (b, i, j) group and a tile along K
    int p = blockIdx.x;             // p indexes the (b, i, j) group
    int k_base = blockIdx.y * TILE_K; // starting index for the current k tile
    int tid = threadIdx.x;          // thread index within the tile
    int k = k_base + tid;           // global k index

    if (p < IJP && k < K) {
        float sum = 0.0f;
        int offsetA = p * L;  // Offset to the A vector for this (b,i,j) group
        int offsetC = p * K;  // Offset to the C vector for this (b,i,j) group
        
        int l = 0;
        // Unroll by factor of 4 for improved performance
        for (; l <= L - 4; l += 4) {
            float a0 = A[offsetA + l];
            float a1 = A[offsetA + l + 1];
            float a2 = A[offsetA + l + 2];
            float a3 = A[offsetA + l + 3];

            sum += a0 * B[(l + 0) * K + k] +
                   a1 * B[(l + 1) * K + k] +
                   a2 * B[(l + 2) * K + k] +
                   a3 * B[(l + 3) * K + k];
        }
        // Process any remaining elements
        for (; l < L; ++l) {
            sum += A[offsetA + l] * B[l * K + k];
        }
        C[offsetC + k] = sum;
    }
}

// The forward function validates inputs, sets up the grid dimensions, and launches the kernel
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

    // Flatten the (BATCH, I, J) dimensions into one index
    int IJP = BATCH * I * J;

    // Grid: one block per (b,i,j) and tile in K
    dim3 blocks(IJP, (K + TILE_K - 1) / TILE_K);
    dim3 threads(TILE_K);

    einsum_kernel_combined<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        IJP, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor multiplication with coalesced memory accesses and loop unrolling (CUDA)");
}
