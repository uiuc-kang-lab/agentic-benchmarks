#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define TILE_SIZE 32
#define MAX_MATRIX_DIM 8192

__constant__ int d_N;

__global__ void triangular_mm_kernel_shared(const float* __restrict__ A,
                                            const float* __restrict__ B,
                                            float* __restrict__ C) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    int ty = threadIdx.y;
    int tx = threadIdx.x;
    
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t <= (row / TILE_SIZE); t++) {
        // Load tile from A and B into shared memory
        if (row < d_N && (t * TILE_SIZE + tx) <= row) {
            s_A[ty][tx] = A[row * d_N + (t * TILE_SIZE + tx)];
        } else {
            s_A[ty][tx] = 0.0f;
        }
        
        if ((t * TILE_SIZE + ty) < d_N && col < d_N) {
            s_B[ty][tx] = B[(t * TILE_SIZE + ty) * d_N + col];
        } else {
            s_B[ty][tx] = 0.0f;
        }
        
        __syncthreads();

        // Compute partial sum for this tile
        if (row < d_N && col <= row) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; k++) {
                if ((t * TILE_SIZE + k) <= row) {
                    sum += s_A[ty][k] * s_B[k][tx];
                }
            }
        }
        
        __syncthreads();
    }

    // Write result
    if (row < d_N && col < d_N) {
        if (col <= row) {
            C[row * d_N + col] = sum;
        } else {
            C[row * d_N + col] = 0.0f;
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
    auto C = torch::empty_like(A);

    // Copy matrix dimension to constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE,
                   (N + TILE_SIZE - 1) / TILE_SIZE);

    // Set shared memory bank size to 8 bytes
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
    
    // Set L1 cache preference to prefer shared memory
    cudaFuncSetCacheConfig(triangular_mm_kernel_shared, cudaFuncCachePreferShared);

    triangular_mm_kernel_shared<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>()
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}