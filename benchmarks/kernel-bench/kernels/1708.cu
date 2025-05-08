#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>

// Define tile size for shared memory tiling
#define TILE_SIZE 32

// Macro to perform asynchronous copy of 4 bytes (one float) from global to shared memory
#define CP_ASYNC_LOAD(dst, src) asm volatile ("cp.async.cg.shared.global [%0], [%1], %2;" :: "r"(dst), "l"(src), "n"(4))

// Device inline functions for min and max
__device__ inline int max_int(int a, int b) { return a > b ? a : b; }
__device__ inline int min_int(int a, int b) { return a < b ? a : b; }

// Pipelined kernel using asynchronous copies (cp.async) and double-buffering in shared memory
__global__ void pipelined_triangular_mm_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                int N) {
    // Global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row >= N || col >= N) return;

    float sum = 0.0f;

    // Allocate shared memory double buffers for tiles of A and B
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    // Number of tiles needed along the k dimension
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    int curr = 0, next = 1;

    // Preload the first tile asynchronously into buffer 'curr'
    if (numTiles > 0) {
        int t = 0;
        int aRow = blockIdx.y * TILE_SIZE + threadIdx.y;
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (aRow < N && aCol < N) {
            CP_ASYNC_LOAD(&As[curr][threadIdx.y][threadIdx.x], &A[aRow * N + aCol]);
        } else {
            As[curr][threadIdx.y][threadIdx.x] = 0.0f;
        }
        int bRow = t * TILE_SIZE + threadIdx.y;
        int bCol = blockIdx.x * TILE_SIZE + threadIdx.x;
        if (bRow < N && bCol < N) {
            CP_ASYNC_LOAD(&Bs[curr][threadIdx.y][threadIdx.x], &B[bRow * N + bCol]);
        } else {
            Bs[curr][threadIdx.y][threadIdx.x] = 0.0f;
        }
        asm volatile("cp.async.commit_group;");
        __syncthreads(); // Ensure the first tile is loaded before computation
    }

    // Loop over all tiles along the k dimension
    for (int t = 0; t < numTiles; t++) {
        // Asynchronously load the next tile into the 'next' buffer, if available
        if (t < numTiles - 1) {
            int next_t = t + 1;
            int aRow = blockIdx.y * TILE_SIZE + threadIdx.y;
            int aCol = next_t * TILE_SIZE + threadIdx.x;
            if (aRow < N && aCol < N) {
                CP_ASYNC_LOAD(&As[next][threadIdx.y][threadIdx.x], &A[aRow * N + aCol]);
            } else {
                As[next][threadIdx.y][threadIdx.x] = 0.0f;
            }
            int bRow = next_t * TILE_SIZE + threadIdx.y;
            int bCol = blockIdx.x * TILE_SIZE + threadIdx.x;
            if (bRow < N && bCol < N) {
                CP_ASYNC_LOAD(&Bs[next][threadIdx.y][threadIdx.x], &B[bRow * N + bCol]);
            } else {
                Bs[next][threadIdx.y][threadIdx.x] = 0.0f;
            }
            asm volatile("cp.async.commit_group;");
        }

        // Wait until the current tile's asynchronous copies are available
        __syncthreads();

        // Compute on the current tile. Only iterate over valid k indexes for triangular matrices.
        int tileStart = t * TILE_SIZE;
        int tileEnd = tileStart + TILE_SIZE;
        int k_start = max_int(tileStart, col);
        int k_end = min_int(tileEnd, row + 1);
        float accum = 0.0f;
        for (int k = k_start; k < k_end; k++) {
            int local_k = k - tileStart; // index in shared tile
            accum += As[curr][threadIdx.y][local_k] * Bs[curr][local_k][threadIdx.x];
        }
        sum += accum;

        __syncthreads(); // Ensure computation is finished before swapping buffers

        // Swap the current and next buffers for double buffering
        int temp = curr;
        curr = next;
        next = temp;
    }

    // Write out the result only for the lower triangular part (row >= col)
    C[row * N + col] = (row >= col) ? sum : 0.0f;
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);
    
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel on the current CUDA stream to overlap async memory transfers and computation
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pipelined_triangular_mm_kernel<<<blocks, threads, 0, stream>>>(
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
    m.def("forward", &forward, "Pipelined Stream Triangular Matrix Multiplication (CUDA)");
}
