#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Compute dot product with shared memory optimization
__device__ __forceinline__ float compute_triangular_dot_product(
    const float* __restrict__ A,
    const float* __restrict__ B,
    const int row,
    const int col,
    const int N,
    float* shared_A,
    float* shared_B,
    const int TILE_SIZE
) {
    float sum = 0.f;
    
    // Process the matrix in tiles
    for (int t = col; t <= row; t += TILE_SIZE) {
        // Load tile into shared memory
        const int tile_end = min(t + TILE_SIZE, row + 1);
        
        // Collaborative loading of tiles
        for (int k = threadIdx.x; k < tile_end - t; k += blockDim.x) {
            shared_A[threadIdx.y * TILE_SIZE + k] = A[row * N + (t + k)];
            shared_B[k * TILE_SIZE + threadIdx.x] = B[(t + k) * N + col];
        }
        __syncthreads();
        
        // Compute dot product for this tile
        #pragma unroll
        for (int k = 0; k < tile_end - t; ++k) {
            sum += shared_A[threadIdx.y * TILE_SIZE + k] * 
                   shared_B[k * TILE_SIZE + threadIdx.x];
        }
        __syncthreads();
    }
    return sum;
}

__global__ void triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N
) {
    const int TILE_SIZE = 32;
    __shared__ float shared_A[TILE_SIZE * TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE * TILE_SIZE];
    
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= N || col >= N) return;

    if (row < col) {
        C[row * N + col] = 0.f;
        return;
    }

    C[row * N + col] = compute_triangular_dot_product(
        A, B, row, col, N, shared_A, shared_B, TILE_SIZE);
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

    const int threads = 32;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

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