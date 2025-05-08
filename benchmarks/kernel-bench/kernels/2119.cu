#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void optimized_tiled_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N
) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Convert thread index to 2D coordinates
    const unsigned int tid = threadIdx.x;
    const unsigned int row = (blockIdx.y * BLOCK_ROWS) + (tid / TILE_DIM);
    const unsigned int col = blockIdx.x * TILE_DIM + (tid % TILE_DIM);
    
    float sum[BLOCK_ROWS] = {0.0f};
    
    // Loop over tiles
    const int numTiles = (N + TILE_DIM - 1) / TILE_DIM;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tile into shared memory with better coalescing
        const int tileOffset = t * TILE_DIM;
        
        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; i++) {
            const int loadRow = row + i;
            if (loadRow < N && tid < TILE_DIM) {
                // Coalesced load for A
                const int srcIdxA = loadRow * N + (tileOffset + tid);
                As[tid][i] = (srcIdxA < N * N && loadRow >= (tileOffset + tid)) ? 
                            __ldg(&A[srcIdxA]) : 0.0f;
                
                // Coalesced load for B
                const int srcIdxB = (tileOffset + tid) * N + col;
                Bs[i][tid] = (srcIdxB < N * N && (tileOffset + tid) >= col) ? 
                            __ldg(&B[srcIdxB]) : 0.0f;
            }
        }
        
        __syncthreads();
        
        // Compute partial sums for each row
        #pragma unroll
        for (int i = 0; i < BLOCK_ROWS; i++) {
            if ((row + i) < N && col < N && (row + i) >= col) {
                #pragma unroll
                for (int k = 0; k < TILE_DIM; k++) {
                    sum[i] += As[k][i] * Bs[k][tid % TILE_DIM];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    #pragma unroll
    for (int i = 0; i < BLOCK_ROWS; i++) {
        const int writeRow = row + i;
        if (writeRow < N && col < N) {
            if (writeRow >= col) {
                C[writeRow * N + col] = sum[i];
            } else {
                C[writeRow * N + col] = 0.0f;
            }
        }
    }
}

at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be same size");

    const int N = static_cast<int>(A.size(0));
    auto C = torch::empty_like(A);

    // Configure grid and block dimensions for better occupancy
    const dim3 block(TILE_DIM * BLOCK_ROWS);  // 1D block configuration
    const dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + BLOCK_ROWS - 1) / BLOCK_ROWS);

    optimized_tiled_triangular_mm_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized tiled triangular matrix multiplication (CUDA)");
}