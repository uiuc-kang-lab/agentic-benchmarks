#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_SIZE 16
#define TILE_DIM (BLOCK_SIZE * 2)

// This kernel computes a 2x2 output block per thread using register tiling.
// The tile loads for matrices A and B are done in a branchless manner by computing safe indices
// and using a mask to zero out out-of-bound elements. This minimizes warp divergence by ensuring
// uniform control flow within warps.
__global__ void branchless_matmul_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int K, int N) {
    // Each thread computes a 2x2 submatrix of C
    int row0 = blockIdx.y * TILE_DIM + threadIdx.y;
    int col0 = blockIdx.x * TILE_DIM + threadIdx.x;
    int row1 = row0 + BLOCK_SIZE;
    int col1 = col0 + BLOCK_SIZE;

    float c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;

    for (int tile = 0; tile < numTiles; tile++) {
        int tStart = tile * TILE_DIM;

        // Load tile from A into shared memory in a branchless way
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int aRow = blockIdx.y * TILE_DIM + threadIdx.y + i * BLOCK_SIZE;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int aCol = tStart + threadIdx.x + j * BLOCK_SIZE;
                int sharedRow = threadIdx.y + i * BLOCK_SIZE;
                int sharedCol = threadIdx.x + j * BLOCK_SIZE;

                // Compute safe indices using min operations to avoid out-of-bound access
                int safe_aRow = (aRow < M) ? aRow : M - 1;
                int safe_aCol = (aCol < K) ? aCol : K - 1;
                float a_val = A[safe_aRow * K + safe_aCol];
                // Use a mask to zero out contributions from out-of-bound indices
                float mask = (aRow < M && aCol < K) ? 1.0f : 0.0f;
                As[sharedRow][sharedCol] = a_val * mask;
            }
        }

        // Load tile from B into shared memory in a branchless way
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int bRow = tStart + threadIdx.y + i * BLOCK_SIZE;
            #pragma unroll
            for (int j = 0; j < 2; j++) {
                int bCol = blockIdx.x * TILE_DIM + threadIdx.x + j * BLOCK_SIZE;
                int sharedRow = threadIdx.y + i * BLOCK_SIZE;
                int sharedCol = threadIdx.x + j * BLOCK_SIZE;

                int safe_bRow = (bRow < K) ? bRow : K - 1;
                int safe_bCol = (bCol < N) ? bCol : N - 1;
                float b_val = B[safe_bRow * N + safe_bCol];
                float mask = (bRow < K && bCol < N) ? 1.0f : 0.0f;
                Bs[sharedRow][sharedCol] = b_val * mask;
            }
        }

        __syncthreads();

        // Compute the partial dot products for the current tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a0 = As[threadIdx.y][k];
            float a1 = As[threadIdx.y + BLOCK_SIZE][k];
            float b0 = Bs[k][threadIdx.x];
            float b1 = Bs[k][threadIdx.x + BLOCK_SIZE];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    // Write the computed 2x2 output block to global memory with boundary checks
    if (row0 < M && col0 < N) {
        C[row0 * N + col0] = c00;
    }
    if (row0 < M && col1 < N) {
        C[row0 * N + col1] = c01;
    }
    if (row1 < M && col0 < N) {
        C[row1 * N + col0] = c10;
    }
    if (row1 < M && col1 < N) {
        C[row1 * N + col1] = c11;
    }
}

// Host function to launch the kernel
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    branchless_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Branchless matrix multiplication with minimized warp divergence (CUDA)");
}
