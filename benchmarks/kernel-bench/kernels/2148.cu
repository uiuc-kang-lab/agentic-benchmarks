#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;

    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    
    // Calculate valid tile range for this thread's position
    int start_tile = col / TILE_SIZE;
    int end_tile = row / TILE_SIZE;

    for (int t = start_tile; t <= end_tile; ++t) {
        // Load A tile with triangular awareness
        int A_col = t * TILE_SIZE + threadIdx.x;
        if (A_col <= row && A_col < N) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * N + A_col];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile with triangular awareness
        int B_row = t * TILE_SIZE + threadIdx.y;
        if (B_row >= col && B_row < N) {
            shared_B[threadIdx.y][threadIdx.x] = B[B_row * N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute only valid k values within this tile
        int k_start = max(t * TILE_SIZE, col);
        int k_end = min((t + 1) * TILE_SIZE - 1, row);
        
        for (int k = k_start; k <= k_end; ++k) {
            int tile_k = k - t * TILE_SIZE;
            sum += shared_A[threadIdx.y][tile_k] * shared_B[tile_k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
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

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, 
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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