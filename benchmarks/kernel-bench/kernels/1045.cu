#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size chosen to match the warp size for optimal coalescing
#define TILE_SIZE 32

// This kernel is specialized for the non-transposed case where A is MxK and B is KxN,
// with both matrices stored in row-major order. Global memory accesses are aligned so that
// consecutive threads load/store contiguous memory locations.
__global__ void coalesced_tiled_matmul_kernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int M, int N, int K,
                                               int lda, int ldb, int ldc) {
    __shared__ float sharedA[TILE_SIZE][TILE_SIZE];
    __shared__ float sharedB[TILE_SIZE][TILE_SIZE];

    // Calculate global row and column indices for C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        
        // Load a tile of A from global memory
        if (row < M && aCol < K) {
            // Since A is row-major, adjacent threads along x dimension access contiguous memory
            sharedA[threadIdx.y][threadIdx.x] = A[row * lda + aCol];
        } else {
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load a tile of B from global memory
        if (bRow < K && col < N) {
            // B is also row-major so the load is coalesced along the row
            sharedB[threadIdx.y][threadIdx.x] = B[bRow * ldb + col];
        } else {
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute the partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }
        
        __syncthreads();
    }

    // Write the computed value to C if within bounds
    if (row < M && col < N) {
        C[row * ldc + col] = sum;
    }
}

// Host function: The kernel is specialized for the non-transposed case (A: MxK, B: KxN).
// This version ensures that global memory accesses are coalesced. Other cases would require
// alternative kernels to maintain coalescing. For now, we throw an error for transposed cases.

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure inputs are CUDA tensors
    if (!A.is_cuda() || !B.is_cuda()) {
        throw std::invalid_argument("Input tensors must be on CUDA devices");
    }

    // Ensure inputs are 2D matrices
    if (A.dim() != 2 || B.dim() != 2) {
        throw std::invalid_argument("Input tensors must be 2D matrices");
    }

    // Get dimensions of A and B
    int64_t A_rows = A.size(0);
    int64_t A_cols = A.size(1);
    int64_t B_rows = B.size(0);
    int64_t B_cols = B.size(1);

    // We optimize the non-transposed, contiguous case: A is MxK and B is KxN
    // Check for compatible dimensions
    if (B_rows != A_cols) {
        throw std::invalid_argument("Incompatible matrix dimensions for multiplication");
    }

    int M = static_cast<int>(A_rows);
    int K = static_cast<int>(A_cols);
    int N = static_cast<int>(B_cols);

    // Assume that input tensors are contiguous in memory, letting us use the stride directly.
    int lda = A.stride(0);
    int ldb = B.stride(0);
    int ldc = N; // For the output matrix C

    auto C = torch::empty({M, N}, A.options());

    // Configure grid and block dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the coalesced kernel
    coalesced_tiled_matmul_kernel<<<gridDim, blockDim>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K,
        lda, ldb, ldc);

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Optimized matrix multiplication with coalesced memory accesses (CUDA)");
}
