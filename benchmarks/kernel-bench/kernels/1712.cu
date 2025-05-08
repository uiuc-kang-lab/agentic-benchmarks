#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// This kernel computes C = tril(A * B) for lower-triangular matrices A and B.
// It uses tiled matrix multiplication in shared memory to ensure coalesced global memory accesses.
// The kernel is optimized for correct and efficient mapping of threads to the problem domain.
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (row >= N || col >= N) return;
    
    float sum = 0.f;
    
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        const int tiled_col = t * TILE_SIZE + threadIdx.x;
        const int tiled_row = t * TILE_SIZE + threadIdx.y;

        // Load data into shared memory
        if (row < N && tiled_col < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + tiled_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && tiled_row < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tiled_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute the sum for the current tile
        if (row >= col) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Write the result
    if (row >= col) {
        C[row * N + col] = sum;
    } else {
        C[row * N + col] = 0.0f;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Input dimensions must match");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Optimized Triangular Matrix Multiplication with Efficient Thread Mapping (CUDA)");
}