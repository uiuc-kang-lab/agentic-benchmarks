#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// Optimized kernel for lower triangular matrix multiplication using warp-level primitives
__global__ void triangular_mm_kernel_warp_optimized(const float* __restrict__ A,
                                                     const float* __restrict__ B,
                                                     float* __restrict__ C,
                                                     int N) {
    // Compute global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Only work on valid indices
    if (row < N && col < N) {
        // For upper-triangular region, set result to zero
        if (row < col) {
            C[row * N + col] = 0.f;
            return;
        }

        float sum = 0.f;

        // Number of tiles along the k-dimension
        int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

        for (int m = 0; m < numTiles; m++) {
            // Calculate the global k-index for loading
            int tile_start = m * TILE_SIZE;

            // Use warp-level primitives to load and compute
            for (int k = 0; k < TILE_SIZE; ++k) {
                int global_k = tile_start + k;
                if (global_k <= row && global_k >= col && global_k < N) {
                    float a_val = A[row * N + global_k];
                    float b_val = B[global_k * N + col];
                    sum += a_val * b_val;
                }
            }
        }

        // Use warp shuffle to reduce within the warp
        unsigned int mask = 0xffffffff;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }

        // Write the result for the first thread in the warp
        if (threadIdx.x % warpSize == 0) {
            C[row * N + col] = sum;
        }
    }
}

// C++ interface exposed to PyTorch
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

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the optimized kernel
    triangular_mm_kernel_warp_optimized<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Warp-optimized Triangular Matrix Multiplication (CUDA)");
}
