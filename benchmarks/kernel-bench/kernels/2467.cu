#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel that overlaps computation with asynchronous global memory transfers
// using cp.async instructions and double-buffered shared memory tiling.

template <typename scalar_t>
__global__ void matmul_transpose_async_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {

    // Tile dimensions
    const int TILE_SIZE = 16;

    // Double buffered shared memory for A and B tiles
    __shared__ scalar_t smemA[2][TILE_SIZE * TILE_SIZE];
    __shared__ scalar_t smemB[2][TILE_SIZE * TILE_SIZE];

    // Compute output indices (using threadIdx.x for row and threadIdx.y for col as in reference)
    int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;

    scalar_t sum = 0;

    // Number of tiles required along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    int currentBuf = 0;  // index for double-buffering

    // ------------------------------------------------------------------
    // Preload the first tile asynchronously into currentBuf if needed
    if (numTiles > 0) {
        int tile = 0;
        int a_index = tile * TILE_SIZE + threadIdx.y;  // for A: index in K dim
        int b_index = tile * TILE_SIZE + threadIdx.x;  // for B: index in K dim
        
        // Asynchronously load one element per thread for A
        if (a_index < K && row < M) {
            asm volatile (
                "cp.async.ca.shared.global [%0], [%1], %2;\n"
                :
                : "r"(&smemA[currentBuf][threadIdx.y * TILE_SIZE + threadIdx.x]),
                  "l"(&A[a_index * M + row]),
                  "n"(sizeof(scalar_t))
            );
        } else {
            smemA[currentBuf][threadIdx.y * TILE_SIZE + threadIdx.x] = 0;
        }

        // Asynchronously load one element per thread for B
        if (b_index < K && col < N) {
            asm volatile (
                "cp.async.ca.shared.global [%0], [%1], %2;\n"
                :
                : "r"(&smemB[currentBuf][threadIdx.y * TILE_SIZE + threadIdx.x]),
                  "l"(&B[col * K + b_index]),
                  "n"(sizeof(scalar_t))
            );
        } else {
            smemB[currentBuf][threadIdx.y * TILE_SIZE + threadIdx.x] = 0;
        }
        // Commit the asynchronous copy group and wait for completion
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Main loop over tiles
    for (int tile = 0; tile < numTiles; tile++) {
        int nextTile = tile + 1;
        int nextBuf = 1 - currentBuf;

        // Prefetch next tile asynchronously into alternate buffer if it exists
        if (nextTile < numTiles) {
            int a_index = nextTile * TILE_SIZE + threadIdx.y;
            int b_index = nextTile * TILE_SIZE + threadIdx.x;
            
            if (a_index < K && row < M) {
                asm volatile (
                    "cp.async.ca.shared.global [%0], [%1], %2;\n"
                    :
                    : "r"(&smemA[nextBuf][threadIdx.y * TILE_SIZE + threadIdx.x]),
                      "l"(&A[a_index * M + row]),
                      "n"(sizeof(scalar_t))
                );
            } else {
                smemA[nextBuf][threadIdx.y * TILE_SIZE + threadIdx.x] = 0;
            }
            
            if (b_index < K && col < N) {
                asm volatile (
                    "cp.async.ca.shared.global [%0], [%1], %2;\n"
                    :
                    : "r"(&smemB[nextBuf][threadIdx.y * TILE_SIZE + threadIdx.x]),
                      "l"(&B[col * K + b_index]),
                      "n"(sizeof(scalar_t))
                );
            } else {
                smemB[nextBuf][threadIdx.y * TILE_SIZE + threadIdx.x] = 0;
            }
            asm volatile("cp.async.commit_group;\n");
        }

        // Wait for asynchronous copies of the current tile to finish
        asm volatile("cp.async.wait_group 0;\n");
        __syncthreads();

        // Compute partial dot product using the tile in the current buffer
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += smemA[currentBuf][i * TILE_SIZE + threadIdx.x] *
                   smemB[currentBuf][threadIdx.y * TILE_SIZE + i];
        }
        __syncthreads();

        // Switch buffers for next iteration
        currentBuf = 1 - currentBuf;
    }

    // Write the computed result to global memory if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// Host function: sets up grid/block dimensions and launches the CUDA kernel

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());
    
    const int TILE_SIZE = 16;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);
                
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_async_kernel", ([&] {
        matmul_transpose_async_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using async cp (CUDA)");
}
