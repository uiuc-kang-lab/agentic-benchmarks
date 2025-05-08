#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int TILE_SIZE=32>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    // Compute row and column indices
    const int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    const int col = blockIdx.y * TILE_SIZE + threadIdx.y;

    // Use double-buffered shared memory with padding to avoid bank conflicts
    __shared__ scalar_t tileA[2][TILE_SIZE][TILE_SIZE + 1];
    __shared__ scalar_t tileB[2][TILE_SIZE][TILE_SIZE + 1];
    
    // Accumulator
    scalar_t sum = 0;

    // Total number of tiles
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Buffer indices for double buffering
    int curr_buffer = 0;
    int next_buffer = 1;

    // Prefetch the first tile into shared memory
    if (numTiles > 0) {
        int k_offset = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        // Use asynchronous copy if available
        scalar_t* destA = &tileA[curr_buffer][threadIdx.y][threadIdx.x];
        const scalar_t* srcA = A + (k_offset + threadIdx.y) * M + row;
        if ((k_offset + threadIdx.y) < K && row < M) {
            asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;" : :
                          "r"(destA), "l"(srcA), "n"(sizeof(scalar_t)));
        } else {
            *destA = 0;
        }

        scalar_t* destB = &tileB[curr_buffer][threadIdx.y][threadIdx.x];
        const scalar_t* srcB = B + col * K + k_offset + threadIdx.x;
        if ((k_offset + threadIdx.x) < K && col < N) {
            asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;" : :
                          "r"(destB), "l"(srcB), "n"(sizeof(scalar_t)));
        } else {
            *destB = 0;
        }
        asm volatile ("cp.async.commit_group;");
#else
        if ((k_offset + threadIdx.y) < K && row < M) {
            tileA[curr_buffer][threadIdx.y][threadIdx.x] = A[(k_offset + threadIdx.y) * M + row];
        } else {
            tileA[curr_buffer][threadIdx.y][threadIdx.x] = 0;
        }
        if ((k_offset + threadIdx.x) < K && col < N) {
            tileB[curr_buffer][threadIdx.y][threadIdx.x] = B[col * K + k_offset + threadIdx.x];
        } else {
            tileB[curr_buffer][threadIdx.y][threadIdx.x] = 0;
        }
#endif
        __syncthreads();
    }

    // Loop over tiles with double buffering
    for (int t = 0; t < numTiles; t++) {
        int k_offset = t * TILE_SIZE;
        int next_t = t + 1;
        
        // Prefetch next tile into next_buffer if available
        if (next_t < numTiles) {
            int k_offset_next = next_t * TILE_SIZE;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            scalar_t* destA = &tileA[next_buffer][threadIdx.y][threadIdx.x];
            const scalar_t* srcA = A + (k_offset_next + threadIdx.y) * M + row;
            if ((k_offset_next + threadIdx.y) < K && row < M) {
                asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;" : :
                              "r"(destA), "l"(srcA), "n"(sizeof(scalar_t)));
            } else {
                *destA = 0;
            }

            scalar_t* destB = &tileB[next_buffer][threadIdx.y][threadIdx.x];
            const scalar_t* srcB = B + col * K + k_offset_next + threadIdx.x;
            if ((k_offset_next + threadIdx.x) < K && col < N) {
                asm volatile ("cp.async.ca.shared.global [%0], [%1], %2;" : :
                              "r"(destB), "l"(srcB), "n"(sizeof(scalar_t)));
            } else {
                *destB = 0;
            }
            asm volatile ("cp.async.commit_group;");
#else
            if ((k_offset_next + threadIdx.y) < K && row < M) {
                tileA[next_buffer][threadIdx.y][threadIdx.x] = A[(k_offset_next + threadIdx.y) * M + row];
            } else {
                tileA[next_buffer][threadIdx.y][threadIdx.x] = 0;
            }
            if ((k_offset_next + threadIdx.x) < K && col < N) {
                tileB[next_buffer][threadIdx.y][threadIdx.x] = B[col * K + k_offset_next + threadIdx.x];
            } else {
                tileB[next_buffer][threadIdx.y][threadIdx.x] = 0;
            }
#endif
        }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
        // Wait for the asynchronous copies of the current tile to complete
        asm volatile ("cp.async.wait_group 0;" ::: "memory");
#endif
        __syncthreads();

        // Compute using the current tile from shared memory
        #pragma unroll 8
        for (int k = 0; k < TILE_SIZE; k += 8) {
            sum += tileA[curr_buffer][k][threadIdx.x] * tileB[curr_buffer][threadIdx.y][k];
            sum += tileA[curr_buffer][k+1][threadIdx.x] * tileB[curr_buffer][threadIdx.y][k+1];
            sum += tileA[curr_buffer][k+2][threadIdx.x] * tileB[curr_buffer][threadIdx.y][k+2];
            sum += tileA[curr_buffer][k+3][threadIdx.x] * tileB[curr_buffer][threadIdx.y][k+3];
            sum += tileA[curr_buffer][k+4][threadIdx.x] * tileB[curr_buffer][threadIdx.y][k+4];
            sum += tileA[curr_buffer][k+5][threadIdx.x] * tileB[curr_buffer][threadIdx.y][k+5];
            sum += tileA[curr_buffer][k+6][threadIdx.x] * tileB[curr_buffer][threadIdx.y][k+6];
            sum += tileA[curr_buffer][k+7][threadIdx.x] * tileB[curr_buffer][threadIdx.y][k+7];
        }

        // Swap buffers for double buffering
        int temp = curr_buffer;
        curr_buffer = next_buffer;
        next_buffer = temp;
    }

    // Write the result back to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    constexpr int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Optimized matrix multiplication with transpose (CUDA)");
}