#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_SIZE 16
#define TILE_DIM 32

__global__ void optimized_tile_warp_matmul_kernel(const float* __restrict__ A, 
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              const int M, const int K, const int N) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_DIM + ty;
    const int col = blockIdx.x * TILE_DIM + tx;
    
    const bool valid_row = row < M;
    const bool valid_col = col < N;
    
    float Cvalues[2][2] = {0.0f};
    
    const int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    
    for (int tile = 0; tile < numTiles; tile++) {
        const int tileOffset = tile * TILE_DIM;
        
        // Load tiles with boundary check and coalesced memory access
        if (valid_row && (tileOffset + tx) < K) {
            As[ty][tx] = A[row * K + tileOffset + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((tileOffset + ty) < K && valid_col) {
            Bs[ty][tx] = B[(tileOffset + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        // Combined register re-use from kernel 1 with efficient warp mechanics from kernel 2
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_val0 = As[ty][k];
            float a_val1 = (ty + BLOCK_SIZE < TILE_DIM) ? As[ty + BLOCK_SIZE][k] : 0.0f;
            float b_val0 = Bs[k][tx];
            float b_val1 = (tx + BLOCK_SIZE < TILE_DIM) ? Bs[k][tx + BLOCK_SIZE] : 0.0f;
            Cvalues[0][0] += a_val0 * b_val0;
            Cvalues[0][1] += a_val0 * b_val1;
            Cvalues[1][0] += a_val1 * b_val0;
            Cvalues[1][1] += a_val1 * b_val1;
        }

        __syncthreads();
    }

    // Write the 2x2 block results to global memory (with boundary checks)
    if (valid_row && valid_col) {
        C[row * N + col] = Cvalues[0][0];
        if (col + BLOCK_SIZE < N) C[row * N + (col + BLOCK_SIZE)] = Cvalues[0][1];
        if (row + BLOCK_SIZE < M) C[(row + BLOCK_SIZE) * N + col] = Cvalues[1][0];
        if ((row + BLOCK_SIZE < M) && (col + BLOCK_SIZE < N)) C[(row + BLOCK_SIZE) * N + (col + BLOCK_SIZE)] = Cvalues[1][1];
    }
}

torch::Tensor optimized_matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    optimized_tile_warp_matmul_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_matmul_cuda, "Optimized tiled and warp-aligned matrix multiplication (CUDA)");
}
