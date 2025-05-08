#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Allow tuning the block dimension. BLOCK_DIM * BLOCK_DIM gives total threads per block.
#ifndef BLOCK_DIM
#define BLOCK_DIM 16  // Experiment with 8, 16, 32, etc. (i.e., 64, 256, 1024 threads per block)
#endif

// CUDA kernel for lower triangular matrix multiplication using shared memory tiling with tunable block size.
// Computes C = tril(A * B) where valid summation indices k are in [col, row] when row >= col.
__global__ void triangular_mm_kernel_tuned(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C,
                                            int N) {
    __shared__ float As[BLOCK_DIM][BLOCK_DIM];
    __shared__ float Bs[BLOCK_DIM][BLOCK_DIM];

    int row = blockIdx.y * BLOCK_DIM + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM + threadIdx.x;

    if (row >= N || col >= N) return;

    float sum = 0.0f;
    int numTiles = (N + BLOCK_DIM - 1) / BLOCK_DIM;
    
    // Iterate over tiles in the k-dimension
    for (int t = 0; t < numTiles; ++t) {
        int tiled_k = t * BLOCK_DIM;
        
        // Load a tile of A: element (row, tiled_k + threadIdx.x)
        if ((tiled_k + threadIdx.x) < N)
            As[threadIdx.y][threadIdx.x] = A[row * N + (tiled_k + threadIdx.x)];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load a tile of B: element (tiled_k + threadIdx.y, col)
        if ((tiled_k + threadIdx.y) < N)
            Bs[threadIdx.y][threadIdx.x] = B[(tiled_k + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        
        // Compute valid k indices for lower triangular multiplication:
        // Valid global k must satisfy: k in [max(tiled_k, col), min(tiled_k+BLOCK_DIM, row+1))
        int global_k_start = tiled_k;
        if (col > global_k_start) global_k_start = col;
        int global_k_end = tiled_k + BLOCK_DIM;
        if (global_k_end > row + 1) global_k_end = row + 1;
        
        for (int k = global_k_start; k < global_k_end; ++k) {
            int k_local = k - tiled_k;
            sum += As[threadIdx.y][k_local] * Bs[k_local][threadIdx.x];
        }
        __syncthreads();
    }
    
    // For elements in the upper triangular part, explicitly set 0.
    if (row < col)
        C[row * N + col] = 0.0f;
    else
        C[row * N + col] = sum;
}

// PyTorch interface: Validates tensors and launches the kernel with a tunable block configuration.
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Inputs must be square matrices");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);
    
    // Experiment with block sizes by tuning the BLOCK_DIM macro.
    dim3 threads(BLOCK_DIM, BLOCK_DIM);
    dim3 blocks((N + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM);
    
    triangular_mm_kernel_tuned<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Tuned Block Size Triangular Matrix Multiplication (CUDA)");
}
