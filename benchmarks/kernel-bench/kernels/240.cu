#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size optimized for occupancy; using 32 to fully utilize shared memory and warp-level parallelism
#define TILE_SIZE 32

// Inline function to perform asynchronous copy of a single float from global memory to shared memory
__device__ inline void cp_async_float(float* dst, const float* src) {
    // Copy 4 bytes (size of float) from src (global) to dst (shared)
    asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;" : : "r"(dst), "l"(src), "n"(4));
}

// Commit the current group of asynchronous copies
__device__ inline void cp_async_commit() {
    asm volatile ("cp.async.commit_group;" :::);
}

// Wait for all asynchronous copies in the group to complete
__device__ inline void cp_async_wait() {
    asm volatile ("cp.async.wait_group 0;" :::);
}

// Kernel implementing batched matrix multiplication with asynchronous double buffering
// This kernel overlaps global memory loads (using cp.async) with computation and minimizes __syncthreads()
// A: (batch_size, M, K), B: (batch_size, K, N), C: (batch_size, M, N)
__global__ void bmm_async_dp_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Declare double-buffered shared memory for tiles of A and B
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    // Compute global indices
    int b = blockIdx.z;  // batch index
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Base pointers for the current batch
    const float* batch_A = A + b * M * K;
    const float* batch_B = B + b * K * N;

    // Compute number of tiles in K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // Double buffering indices
    int curr = 0;
    int next = 1;

    // Preload the first tile (t = 0) into the current buffer asynchronously
    int t = 0;
    int aCol = t * TILE_SIZE + threadIdx.x;
    int bRow = t * TILE_SIZE + threadIdx.y;
    if (row < M && aCol < K) {
        cp_async_float(&As[curr][threadIdx.y][threadIdx.x], batch_A + row * K + aCol);
    } else {
        As[curr][threadIdx.y][threadIdx.x] = 0.0f;
    }
    if (bRow < K && col < N) {
        cp_async_float(&Bs[curr][threadIdx.y][threadIdx.x], batch_B + bRow * N + col);
    } else {
        Bs[curr][threadIdx.y][threadIdx.x] = 0.0f;
    }
    cp_async_commit();
    cp_async_wait();
    __syncthreads();  // Ensure the first tile is loaded into shared memory

    // Loop over all tiles in the K dimension
    for (t = 0; t < numTiles; t++) {
        // If there is a next tile, begin asynchronous copy into the alternate buffer
        if (t < numTiles - 1) {
            int next_aCol = (t + 1) * TILE_SIZE + threadIdx.x;
            int next_bRow = (t + 1) * TILE_SIZE + threadIdx.y;
            if (row < M && next_aCol < K) {
                cp_async_float(&As[next][threadIdx.y][threadIdx.x], batch_A + row * K + next_aCol);
            } else {
                As[next][threadIdx.y][threadIdx.x] = 0.0f;
            }
            if (next_bRow < K && col < N) {
                cp_async_float(&Bs[next][threadIdx.y][threadIdx.x], batch_B + next_bRow * N + col);
            } else {
                Bs[next][threadIdx.y][threadIdx.x] = 0.0f;
            }
            cp_async_commit();
        }

        // Multiply the elements of the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[curr][threadIdx.y][k] * Bs[curr][k][threadIdx.x];
        }

        // For all but the last tile, wait for the next tile's data and swap buffers
        if (t < numTiles - 1) {
            cp_async_wait();
            __syncthreads();  // Ensure the newly loaded tile in the alternate buffer is ready for use
            int temp = curr;
            curr = next;
            next = temp;
        }
    }

    // Write the computed result back to global memory
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
    }
}

// Host function to launch the kernel
torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    // Configure grid and block dimensions
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE, batch_size);

    bmm_async_dp_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with asynchronous double buffering and minimal synchronization (CUDA)");
}
