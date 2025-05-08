#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use double-buffering and asynchronous copies (cp.async) for overlapping global loads with computation

template <typename scalar_t>
__global__ void matmul_transpose_shared_unroll_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    const int TILE_SIZE = 32;
    // Double-buffered shared memory; padding to avoid bank conflicts
    __shared__ scalar_t A_shared[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t B_shared[2][TILE_SIZE][TILE_SIZE + 1];

    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    scalar_t sum = 0;
    
    // Total number of tiles
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Preload the first tile synchronously into buffer 0
    if (row < M && (0 * TILE_SIZE + ty) < K) {
        A_shared[0][ty][tx] = A[row + (0 * TILE_SIZE + ty) * M];
    } else {
        A_shared[0][ty][tx] = 0;
    }
    if (col < N && (0 * TILE_SIZE + tx) < K) {
        B_shared[0][tx][ty] = B[col * K + 0 * TILE_SIZE + tx];
    } else {
        B_shared[0][tx][ty] = 0;
    }
    
    // Ensure the first tile is loaded
    __syncthreads();

    // Loop over all tiles
    for (int tile = 0; tile < numTiles; ++tile) {
        int curr = tile & 1;
        int next = (tile + 1) & 1;

        // Asynchronously prefetch next tile if it exists
        if (tile < numTiles - 1) {
            // Load A tile asynchronously using cp.async if supported.
            // Note: cp.async transfers 16 bytes at a time, here we assume scalar_t is float (4 bytes), so one element may be used per thread.
            if (row < M && ((tile + 1) * TILE_SIZE + ty) < K) {
                asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;"
                              :
                              : "r"(&A_shared[next][ty][tx]),
                                "l"(A + row + (((tile + 1) * TILE_SIZE + ty)) * M),
                                "n"(sizeof(scalar_t)));
            } else {
                A_shared[next][ty][tx] = 0;
            }
            if (col < N && ((tile + 1) * TILE_SIZE + tx) < K) {
                asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;"
                              :
                              : "r"(&B_shared[next][tx][ty]),
                                "l"(B + col * K + (tile + 1) * TILE_SIZE + tx),
                                "n"(sizeof(scalar_t)));
            } else {
                B_shared[next][tx][ty] = 0;
            }
        }

        // Make sure asynchronous copies complete before using data
        // cp.async.wait_group 0 waits for all issued cp.async operations
        asm volatile ("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();

        // Compute on the current tile
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                if ((tile * TILE_SIZE + k) < K) {
                    sum += A_shared[curr][k][tx] * B_shared[curr][k][ty];
                }
            }
        }

        // Synchronize to ensure next tile is ready before next iteration
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    const int TILE_SIZE = 32;
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE + 1];
    
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    scalar_t sum = 0;
    
    #pragma unroll 4
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        if (row < M && (tile * TILE_SIZE + ty) < K) {
            A_shared[ty][tx] = A[row + (tile * TILE_SIZE + ty) * M];
        } else {
            A_shared[ty][tx] = 0;
        }
        
        if (col < N && (tile * TILE_SIZE + tx) < K) {
            B_shared[tx][ty] = B[col * K + tile * TILE_SIZE + tx];
        } else {
            B_shared[tx][ty] = 0;
        }
        
        __syncthreads();
        
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                if ((tile * TILE_SIZE + k) < K) {
                    sum += A_shared[k][tx] * B_shared[k][ty];
                }
            }
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    const int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_shared_unroll_kernel", ([&] {
        matmul_transpose_shared_unroll_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Optimized matrix multiplication with transposed matrices using shared memory and loop unrolling");
}