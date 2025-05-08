#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32

// This kernel uses shared memory tiling and leverages __ldg() to perform read-only loads from global memory.
// It assumes that the input matrices A and B are aligned to 128-bit boundaries, which enables optimal vectorized loads under the hood.

template <typename scalar_t>
__global__ void ldg_matmul_kernel(const scalar_t* __restrict__ A,
                                   const scalar_t* __restrict__ B,
                                   scalar_t* __restrict__ C,
                                   int M, int K, int N) {
    // Using double-buffering in shared memory to overlap computation with global loads
    __shared__ scalar_t sA[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[2][TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t value = 0;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;

    // Preload the first tile into buffer 0
    int buf = 0;
    {
        int aCol = 0 * TILE_WIDTH + threadIdx.x;
        int bRow = 0 * TILE_WIDTH + threadIdx.y;
        if (row < M && aCol < K)
            sA[buf][threadIdx.y][threadIdx.x] = __ldg(&A[row * K + aCol]);
        else
            sA[buf][threadIdx.y][threadIdx.x] = 0;

        if (bRow < K && col < N)
            sB[buf][threadIdx.y][threadIdx.x] = __ldg(&B[bRow * N + col]);
        else
            sB[buf][threadIdx.y][threadIdx.x] = 0;
    }
    __syncthreads();

    // Loop over tiles with double buffering to overlap loads and computation
    for (int t = 0; t < numTiles - 1; t++) {
        int curr = t & 1;
        int next = 1 - curr;

        // Preload the next tile into the alternate buffer
        int aColNext = (t + 1) * TILE_WIDTH + threadIdx.x;
        int bRowNext = (t + 1) * TILE_WIDTH + threadIdx.y;
        if (row < M && aColNext < K)
            sA[next][threadIdx.y][threadIdx.x] = __ldg(&A[row * K + aColNext]);
        else
            sA[next][threadIdx.y][threadIdx.x] = 0;

        if (bRowNext < K && col < N)
            sB[next][threadIdx.y][threadIdx.x] = __ldg(&B[bRowNext * N + col]);
        else
            sB[next][threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        // Compute multiplication for the current tile
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            // Cache the shared memory loads in registers to reduce repeated accesses
            scalar_t a_val = sA[curr][threadIdx.y][i];
            scalar_t b_val = sB[curr][i][threadIdx.x];
            value += a_val * b_val;
        }
        __syncthreads();
    }

    // Process the last tile
    int last = (numTiles - 1) & 1;
    #pragma unroll
    for (int i = 0; i < TILE_WIDTH; i++) {
        scalar_t a_val = sA[last][threadIdx.y][i];
        scalar_t b_val = sB[last][i][threadIdx.x];
        value += a_val * b_val;
    }

    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// Forward function
torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "ldg_matmul_kernel", ([&] {
        ldg_matmul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

// Binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "LDG optimized matrix multiplication (CUDA)");
}
