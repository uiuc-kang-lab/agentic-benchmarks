#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void coalesced_triangular_mm_kernel(const float* __restrict__ A,
                                                const float* __restrict__ B,
                                                float* __restrict__ C,
                                                int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE+1];  // Padding for avoiding bank conflict
    __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // All threads in a warp access consecutive memory addresses
    int aligned_row = (row / TILE_SIZE) * TILE_SIZE;
    int aligned_col = (col / TILE_SIZE) * TILE_SIZE;

    if (row >= N || col >= N) return;

    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; ++t) {
        int tile_start = t * TILE_SIZE;
        
        // Coalesced memory access pattern
        if (tile_start + threadIdx.x <= row) {
            As[threadIdx.y][threadIdx.x] = A[row * N + tile_start + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (tile_start + threadIdx.y < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(tile_start + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        int k_start = max(tile_start, col);
        int k_end = min(tile_start + TILE_SIZE, row + 1);

        for (int k = k_start; k < k_end; ++k) {
            int k_tile = k - tile_start;
            sum += As[threadIdx.y][k_tile] * Bs[k_tile][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);
    
    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (N + TILE_SIZE - 1) / TILE_SIZE);

    coalesced_triangular_mm_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                                   B.data_ptr<float>(),
                                                   C.data_ptr<float>(),
                                                   N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced lower triangular matrix multiplication (CUDA)");
}