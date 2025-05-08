#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define TILE_DIM 16

__global__ void triangular_mm_kernel_const(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N) {
    
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    
    // Only compute if we're in the lower triangular part
    if (row >= col && row < N && col < N) {
        // Loop over tiles
        for (int t = 0; t < (N + TILE_DIM - 1) / TILE_DIM; ++t) {
            // Load tile from A and B into shared memory
            int tileStart = t * TILE_DIM;
            if (tileStart + threadIdx.x <= row && tileStart + threadIdx.y < N) {
                As[threadIdx.y][threadIdx.x] = A[row * N + (tileStart + threadIdx.x)];
                Bs[threadIdx.y][threadIdx.x] = B[(tileStart + threadIdx.y) * N + col];
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial sum for this tile
            for (int k = 0; k < TILE_DIM; ++k) {
                if (tileStart + k <= row && tileStart + k >= col) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
            
            __syncthreads();
        }
        
        C[row * N + col] = sum;
    } else if (row < col && row < N && col < N) {
        // Upper triangular part should be zero
        C[row * N + col] = 0.0f;
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
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds constant memory limit");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Copy input matrices to constant memory
    cudaMemcpyToSymbol(A_const, A.data_ptr<float>(), N * N * sizeof(float));
    cudaMemcpyToSymbol(B_const, B.data_ptr<float>(), N * N * sizeof(float));

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    triangular_mm_kernel_const<<<numBlocks, threadsPerBlock>>>(
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