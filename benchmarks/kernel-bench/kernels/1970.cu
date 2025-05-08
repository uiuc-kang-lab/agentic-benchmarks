#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Kernel optimized with warp-level primitives for reduction
__global__ void warp_optimized_triangular_mm(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row >= N || col >= N) return;

    // Enforce lower triangular result
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int m = 0; m < numTiles; m++) {
        int kA = m * TILE_SIZE + threadIdx.x;
        int kB = m * TILE_SIZE + threadIdx.y;

        // Load tiles with bounds checking
        sA[threadIdx.y][threadIdx.x] = (kA < N && row >= kA) ? A[row * N + kA] : 0.0f;
        sB[threadIdx.y][threadIdx.x] = (kB < N && kB >= col) ? B[kB * N + col] : 0.0f;
        
        __syncthreads();

        // Compute valid k range for this tile
        int kStart = max(col, m * TILE_SIZE);
        int kEnd = min(row + 1, (m + 1) * TILE_SIZE);
        
        #pragma unroll
        for (int k = kStart - m * TILE_SIZE; k < kEnd - m * TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // Write the result for the first thread in the warp
    if (threadIdx.x % warpSize == 0) {
        C[row * N + col] = sum;
    }
}

// PyTorch interface function
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square matrices");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    warp_optimized_triangular_mm<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Warp Optimized Triangular Matrix Multiplication (CUDA)");
}
