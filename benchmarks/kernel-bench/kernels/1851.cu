#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define MAX_MATRIX_DIM 8192  // Maximum supported matrix dimension

// Declare constant memory
__constant__ int d_N;  // Matrix dimension in constant memory

__global__ void triangular_mm_kernel_shared_warp(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C) {
    __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.f;

    for (int tile = 0; tile < (d_N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Load tiles into shared memory
        if (row < d_N && tile * BLOCK_SIZE + threadIdx.x < d_N) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * d_N + tile * BLOCK_SIZE + threadIdx.x];
        } else {
            shared_A[threadIdx.y][threadIdx.x] = 0.f;
        }

        if (col < d_N && tile * BLOCK_SIZE + threadIdx.y < d_N) {
            shared_B[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * d_N + col];
        } else {
            shared_B[threadIdx.y][threadIdx.x] = 0.f;
        }

        __syncthreads();

        // Compute partial product for this tile
        if (row < d_N && col <= row) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
            }
        }

        __syncthreads();
    }

    // Use warp-level primitives to finalize reduction
    if (row < d_N && col <= row) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (threadIdx.x % warpSize == 0) {
            C[row * d_N + col] = sum;
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

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Set L1 cache preference to prefer shared memory
    cudaFuncSetCacheConfig(triangular_mm_kernel_shared_warp, cudaFuncCachePreferShared);

    triangular_mm_kernel_shared_warp<<<numBlocks, threadsPerBlock>>>(
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
