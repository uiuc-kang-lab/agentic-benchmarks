#include <torch/extension.h>
#include <cuda_runtime.h>

// Kernel to compute: C[b,i,j,k] = sum_{l=0}^{L-1} A[b,i,j,l] * B[l,k]
// This kernel uses shared memory to preload the A vector for each (b,i,j) and tiles the L dimension to also load portions of B into shared memory,
// reducing global memory latency. Proper __syncthreads are used to avoid race conditions.

// grid configuration:
//   Each block is assigned to one tile for a given (b,i,j) and a tile of the K dimension.
//   Let num_k_tiles = ceil(K / tileK). Then gridDim.x = (BATCH * I * J) * num_k_tiles.
// Block index decoding:
//   outer_index = blockIdx.x / num_k_tiles for the (b,i,j) triple, and k_tile = blockIdx.x % num_k_tiles.
// Threads with threadIdx.x < tileK compute output results, while all threads help in loading shared memory.

__global__ void einsum_kernel_tiled(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int BATCH,
    int I,
    int J,
    int L,
    int K,
    int tileL,
    int tileK
) {
    // Calculate number of k tiles
    int numKTiles = (K + tileK - 1) / tileK;

    // Decode block index
    int outer_index = blockIdx.x / numKTiles;  // corresponds to a unique (b, i, j)
    int k_tile_idx = blockIdx.x % numKTiles;     // which tile of K this block handles

    // Decode b, i, j from outer_index
    int bij = outer_index; // index in [0, BATCH * I * J)
    int b = bij / (I * J);
    int rem = bij % (I * J);
    int i = rem / J;
    int j = rem % J;

    // Compute k starting index for this block
    int k_start = k_tile_idx * tileK;

    // Allocate shared memory:
    // First: shared_A for the vector A[b,i,j,*] of size L
    // Second: shared_B for a tile of B of size tileL x tileK
    extern __shared__ float sharedMem[];
    float* shared_A = sharedMem;           // size L floats
    float* shared_B = sharedMem + L;         // size tileL * tileK floats

    int tid = threadIdx.x;
    
    // Load the entire A[b,i,j,*] vector into shared memory cooperatively
    for (int idx = tid; idx < L; idx += blockDim.x) {
        int a_index = b * (I * J * L) + i * (J * L) + j * L + idx;
        shared_A[idx] = A[a_index];
    }
    __syncthreads();

    // We'll use threads with tid < tileK for computing the output in the k-dimension
    float sum = 0.0f;

    // Loop over the L dimension in tiles
    for (int l_tile = 0; l_tile < L; l_tile += tileL) {
        // Load a tile of B from global memory into shared memory
        // The tile covers rows [l_tile, l_tile + tileL) and columns [k_start, k_start + tileK)
        int numElements = tileL * tileK;
        for (int t = tid; t < numElements; t += blockDim.x) {
            int row = t / tileK;  // index within the tile along L
            int col = t % tileK;  // index within the tile along K
            int l_global = l_tile + row;
            int k_global = k_start + col;
            float val = 0.0f;
            if (l_global < L && k_global < K) {
                val = B[l_global * K + k_global];
            }
            shared_B[t] = val;
        }
        __syncthreads();

        // Determine effective tile height (for boundary conditions)
        int effectiveTileL = tileL;
        if (l_tile + tileL > L) {
            effectiveTileL = L - l_tile;
        }

        // Threads with tid < tileK compute the dot product for the elements in this k tile
        if (tid < tileK) {
            for (int m = 0; m < effectiveTileL; m++) {
                sum += shared_A[l_tile + m] * shared_B[m * tileK + tid];
            }
        }
        __syncthreads();
    }

    // Write the computed result back to global memory if within bounds
    if (tid < tileK) {
        int k_global = k_start + tid;
        if (k_global < K) {
            int c_index = b * (I * J * K) + i * (J * K) + j * K + k_global;
            C[c_index] = sum;
        }
    }
}

// Forward function accessible from PyTorch

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 4, "Tensor A must be 4D");
    TORCH_CHECK(B.dim() == 2, "Tensor B must be 2D");
    TORCH_CHECK(A.size(3) == B.size(0), "A.size(3) must equal B.size(0)");

    int BATCH = A.size(0);
    int I = A.size(1);
    int J = A.size(2);
    int L = A.size(3);
    int K = B.size(1);

    auto C = torch::zeros({BATCH, I, J, K}, A.options());

    // Tiling parameters
    int tileL = 32; // tile size for the L dimension
    int tileK = 32; // tile size for the K dimension

    int numKTiles = (K + tileK - 1) / tileK;
    // Each block processes one (b,i,j) and one tile of the K dimension
    int numBlocks = BATCH * I * J * numKTiles;

    int threads = 256;  // number of threads per block

    // Shared memory: space for A vector (L floats) + space for B tile (tileL * tileK floats)
    size_t sharedBytes = L * sizeof(float) + tileL * tileK * sizeof(float);

    einsum_kernel_tiled<<<numBlocks, threads, sharedBytes>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        BATCH, I, J, L, K,
        tileL, tileK
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "4D tensor-matrix multiplication with shared memory tiling (CUDA)");
}
