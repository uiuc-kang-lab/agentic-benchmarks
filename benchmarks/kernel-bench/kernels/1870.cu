#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts
    __shared__ float Bs[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    const int block_row = blockIdx.y * BLOCK_ROWS;
    const int row = block_row + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Collaborative loading of A and B tiles into shared memory
        const int tile_idx = t * TILE_DIM;
        
        // Load tile of A - coalesced access
        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; i++) {
            if ((block_row + i) < N && (tile_idx + threadIdx.x) < N) {
                As[threadIdx.y + i][threadIdx.x] = A[(block_row + i) * N + tile_idx + threadIdx.x];
            } else {
                As[threadIdx.y + i][threadIdx.x] = 0.0f;
            }
        }

        // Load tile of B - coalesced access
        if ((tile_idx + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile_idx + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial results
        if (row < N && col < N && row >= col) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; ++k) {
                const int global_k = tile_idx + k;
                if (global_k >= col && global_k <= row) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
        }

        __syncthreads();
    }

    // Write results
    if (row < N && col < N) {
        if (row >= col) {
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_DIM, BLOCK_ROWS);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + BLOCK_ROWS - 1) / BLOCK_ROWS);

    // Set L1 cache preference to prefer shared memory
    cudaFuncSetCacheConfig(triangular_mm_kernel, cudaFuncCachePreferShared);

    triangular_mm_kernel<<<grid, threads>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}