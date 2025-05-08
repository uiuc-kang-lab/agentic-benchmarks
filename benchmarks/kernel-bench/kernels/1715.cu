#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define MAX_MATRIX_DIM 1024

__constant__ float d_A[TILE_SIZE * MAX_MATRIX_DIM];
__constant__ float d_B[TILE_SIZE * MAX_MATRIX_DIM];

__global__ void triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N,
    const int tile_idx) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            // Calculate starting point for this tile
            int start_k = max(col, tile_idx * TILE_SIZE);
            int end_k = min(row, (tile_idx + 1) * TILE_SIZE - 1);
            
            // Only process if we're in a valid range
            if (start_k <= end_k) {
                for (int k = start_k; k <= end_k; ++k) {
                    // Use constant memory for the current tile
                    int local_k = k - tile_idx * TILE_SIZE;
                    sum += d_A[row * TILE_SIZE + local_k] * 
                           d_B[k * N + col];
                }
                atomicAdd(&C[row * N + col], sum);
            }
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
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    auto C = torch::zeros_like(A);

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    // Process matrix in tiles
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < num_tiles; ++tile) {
        // Copy tile to constant memory
        int tile_start = tile * TILE_SIZE;
        int tile_size = min(TILE_SIZE, N - tile_start);
        
        cudaMemcpyToSymbol(d_A, 
                           A.data_ptr<float>() + tile_start, 
                           tile_size * N * sizeof(float));
        cudaMemcpyToSymbol(d_B,
                           B.data_ptr<float>() + tile_start * N,
                           tile_size * N * sizeof(float));

        triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            N,
            tile
        );
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}