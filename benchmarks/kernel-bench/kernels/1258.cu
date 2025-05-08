/*
This CUDA kernel combines the coalesced memory access strategy from Kernel 1 and the multiple-items-per-thread striding from Kernel 2.
Each block processes a specific (batch, i, j) group and a tile in the K dimension. Within each block, each thread computes multiple dot-product results along K (with ITEMS_PER_THREAD outputs per thread) with stride, ensuring that global memory accesses for B and C are coalesced.

The kernel uses loop unrolling for the inner accumulation loop to improve throughput. This approach reduces memory bandwidth overhead and increases arithmetic intensity.
*/

#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ITEMS_PER_THREAD 4

// Combined kernel: for each p (flattened batch, i, j), and for each k in tile, compute:
// C[p,k] = sum_{l=0}^{L-1} A[p, l] * B[l, k]
// Grid: gridDim.x = IJP (number of groups); gridDim.y = ceil(K / (BLOCK_SIZE * ITEMS_PER_THREAD))
// Each thread handles ITEMS_PER_THREAD outputs in the k dimension with a stride of BLOCK_SIZE.

__global__ void einsum_kernel_combined(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int IJP,  // Total number of groups (BATCH * I * J)
    int L,
    int K
) {
    // Shared memory for B matrix tile
    __shared__ float B_shared[32][BLOCK_SIZE];  // 32 is the tile size for L dimension
    
    // p indexes the (batch, i, j) group
    int p = blockIdx.x;
    // Each block processes a tile in the K dimension
    int tile_idx = blockIdx.y;
    int tid = threadIdx.x;

    // Each thread processes ITEMS_PER_THREAD elements with stride BLOCK_SIZE across the k dimension
    int k_base = tile_idx * (BLOCK_SIZE * ITEMS_PER_THREAD) + tid;

    int offsetA = p * L;  // A is [IJP, L]
    int offsetC = p * K;  // C is [IJP, K]

    // Initialize accumulators for each of the ITEMS_PER_THREAD k values
    float sums[ITEMS_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        sums[i] = 0.0f;
    }

    // Process L dimension in tiles of size 32
    for (int l_tile = 0; l_tile < L; l_tile += 32) {
        // Cooperatively load B tile into shared memory
        #pragma unroll
        for (int i = 0; i < 32; i += BLOCK_SIZE/32) {
            int l_local = i + (tid % 32);
            int k_local = tid / 32;
            if ((l_tile + l_local) < L && k_base < K) {
                B_shared[l_local][k_local] = B[(l_tile + l_local) * K + k_base];
            }
        }
        __syncthreads();

        // Process the tile
        #pragma unroll
        for (int l_local = 0; l_local < min(32, L - l_tile); ++l_local) {
            float a_val = A[offsetA + l_tile + l_local];
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_THREAD; i++) {
                int k = k_base + i * BLOCK_SIZE;
                if (k < K) {
                    sums[i] += a_val * B_shared[l_local][tid];
                }
            }
        }
        __syncthreads();
    }

    // Write the computed results to C with boundary checking
    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; i++) {
        int k = k_base + i * BLOCK_SIZE;
        if (k < K) {
            C[offsetC + k] = sums[i];
        }
    }
}

// The forward function sets up grid dimensions and launches the combined kernel

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

    // Calculate the number of tiles in the K dimension
    int k_tiles = (K + (BLOCK_SIZE * ITEMS_PER_THREAD) - 1) / (BLOCK_SIZE * ITEMS_PER_THREAD);
    dim3 blocks(IJP, k_tiles);
    dim3 threads(BLOCK_SIZE);

    einsum_kernel_combined<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        IJP, L, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 4D tensor-matrix multiplication combining coalesced and strided accesses (CUDA)");
}
