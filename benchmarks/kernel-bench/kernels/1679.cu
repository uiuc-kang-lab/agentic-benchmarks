#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// This kernel assigns each lower-triangular output element (i,j) to a single warp.
// The warp cooperatively computes the reduction over k from j to i using a tiling
// strategy with shared memory to cache portions of A[i,k] and B[k,j].
// Intra-tile partial products are reduced using warp-level primitives (__shfl_down_sync).
// Only the first lane of each warp writes the final sum to the output matrix.
__global__ void triangular_mm_kernel_shared(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    const int warpSize = 32;
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int warpId = globalThreadId / warpSize;
    int lane = globalThreadId % warpSize;

    // Total number of lower-triangular elements
    int total_elements = N * (N + 1) / 2;
    if (warpId >= total_elements) return;

    // Map warpId to matrix indices (i,j) for the lower-triangular part
    // i = floor((sqrt(8*warpId+1)-1)/2) and j = warpId - i*(i+1)/2
    int i = (int)floor((sqrtf(8.0f * warpId + 1.0f) - 1.0f) * 0.5f);
    int tri_offset = i * (i + 1) / 2;
    int j = warpId - tri_offset;

    float sum = 0.0f;
    const int tile_size = 32; // tile dimension for the reduction

    // Each warp in a block will get its own slice of shared memory.
    // Compute warp index within the block
    int warpInBlock = threadIdx.x / warpSize;
    // Shared memory is partitioned per warp: first half for A-tile, second half for B-tile
    extern __shared__ float shared[];
    float* aTile = shared + warpInBlock * tile_size;
    float* bTile = shared + (blockDim.x / warpSize) * tile_size + warpInBlock * tile_size;

    // Loop over k in tiles from j to i
    for (int k_start = j; k_start <= i; k_start += tile_size) {
        int current_tile_size = (i - k_start + 1 < tile_size) ? (i - k_start + 1) : tile_size;
        
        // Each warp loads a tile of A[i, k] and B[k, j] into its private shared memory region
        if (lane < current_tile_size) {
            aTile[lane] = A[i * N + (k_start + lane)];
            bTile[lane] = B[(k_start + lane) * N + j];
        }
        // Synchronize lanes in the warp to ensure tile is loaded
        __syncwarp();
        
        // Each lane processes a subset of the tile in stride
        float tile_sum = 0.0f;
        for (int t = lane; t < current_tile_size; t += warpSize) {
            tile_sum += aTile[t] * bTile[t];
        }
        
        // Reduce the tile_sum across the warp using warp shuffle reduction
        unsigned int mask = 0xffffffff;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            tile_sum += __shfl_down_sync(mask, tile_sum, offset);
        }
        // Only lane 0 adds the reduced tile sum to the global sum
        if (lane == 0) {
            sum += tile_sum;
        }
        // Synchronize warp before loading the next tile
        __syncwarp();
    }

    // Broadcast the final sum from lane 0 to all threads in the warp
    sum = __shfl_sync(0xffffffff, sum, 0);
    if (lane == 0) {
        C[i * N + j] = sum;
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    // Allocate output tensor and initialize with zeros to cover the upper-triangular part
    auto C = torch::zeros_like(A);

    int total_elements = N * (N + 1) / 2; // number of lower-triangular elements
    const int warpSize = 32;
    // Each lower-triangular element will be computed by one warp
    int totalThreads = total_elements * warpSize;
    // Choose block configuration; blockDim must be a multiple of warpSize (e.g., 128 threads per block)
    int threadsPerBlock = 128;
    int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

    // Compute shared memory size per block:
    // Each block has (threadsPerBlock/warpSize) warps, and each warp uses 2 * tile_size floats
    const int tile_size = 32;
    size_t sharedMemBytes = (threadsPerBlock / warpSize) * tile_size * 2 * sizeof(float);

    triangular_mm_kernel_shared<<<numBlocks, threadsPerBlock, sharedMemBytes>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication with shared memory and warp-level reduction (CUDA)");
}
