#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_SIZE 16
#define TILE_DIM 32

__global__ void warp_tile_optimized_matmul_kernel(const float* __restrict__ A, 
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
        
        // Load tiles with boundary check and vectorized loads
        for(int i = 0; i < 2; i++) {
            if (valid_row && (tileOffset + tx) < K) {
                As[ty + i * BLOCK_SIZE][tx] = A[(row + i * BLOCK_SIZE) * K + tileOffset + tx];
            } else {
                As[ty + i * BLOCK_SIZE][tx] = 0.0f;
            }

            if ((tileOffset + ty) < K && valid_col) {
                Bs[ty][tx + i * BLOCK_SIZE] = B[(tileOffset + ty) * N + col + i * BLOCK_SIZE];
            } else {
                Bs[ty][tx + i * BLOCK_SIZE] = 0.0f;
            }
        }
        __syncthreads();

        // Compute using 2x2 block register accumulation
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_val0 = As[ty][k];
            float a_val1 = As[ty + BLOCK_SIZE][k];
            float b_val0 = Bs[k][tx];
            float b_val1 = Bs[k][tx + BLOCK_SIZE];
            Cvalues[0][0] += a_val0 * b_val0;
            Cvalues[0][1] += a_val0 * b_val1;
            Cvalues[1][0] += a_val1 * b_val0;
            Cvalues[1][1] += a_val1 * b_val1;
        }
        __syncthreads();
    }

    if (valid_row && valid_col) {
        for(int i = 0; i < 2; i++) {
            for(int j = 0; j < 2; j++) {
                int outputRow = row + i * BLOCK_SIZE;
                int outputCol = col + j * BLOCK_SIZE;
                if (outputRow < M && outputCol < N) {
                    C[outputRow * N + outputCol] = Cvalues[i][j];
                }
            }
        }
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
    
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    auto C = torch::zeros({M, N}, A.options());
    
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    warp_tile_optimized_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Warp-tiled optimized matrix multiplication (CUDA)");
}
