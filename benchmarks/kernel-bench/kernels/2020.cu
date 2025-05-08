#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// This CUDA kernel performs lower triangular matrix multiplication, C = A * B, for lower triangular matrices A and B.
// It uses a standard tiled matrix multiplication approach with shared memory to ensure memory coalescing.
// Global memory accesses to A are coalesced because consecutive threads in a warp load contiguous elements from a row,
// and accesses to B are coalesced because each thread loads an element from a row of B (with consecutive columns accessed by adjacent threads).
// In addition, the kernel masks out elements outside the effective triangular region (i.e. A[i,j] is only used when j <= i
// and B[i,j] is only used when i >= j) ensuring the correct result for lower triangular matrices.

__global__ void coalesced_triangular_tiled_kernel(const float* __restrict__ A,
                                                   const float* __restrict__ B,
                                                   float* __restrict__ C,
                                                   int N) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int m = 0; m < numTiles; m++) {
        int colA = m * TILE_SIZE + tx;            // Column index for A
        // Load tile of A: only valid if colA is within bounds and in the lower triangular region (colA <= row)
        if (row < N && colA < N && (colA <= row))
            sA[ty][tx] = A[row * N + colA];
        else
            sA[ty][tx] = 0.0f;

        int rowB = m * TILE_SIZE + ty;             // Row index for B
        // Load tile of B: only valid if rowB is in bounds and in the lower triangular region (rowB >= col)
        if (rowB < N && col < N && (rowB >= col))
            sB[ty][tx] = B[rowB * N + col];
        else
            sB[ty][tx] = 0.0f;

        __syncthreads();
        
        // Multiply the two tiles together
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }
    
    // Write the output only for the lower triangular part
    if (row < N && col < N) {
        C[row * N + col] = (row >= col) ? sum : 0.0f;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    coalesced_triangular_tiled_kernel<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Coalesced Tiled Lower Triangular Matrix Multiplication (CUDA)");
}
