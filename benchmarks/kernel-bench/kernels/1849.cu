#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define MAX_MATRIX_DIM 8192  // Maximum supported matrix dimension

// Declare constant memory
__constant__ int d_N;  // Matrix dimension in constant memory
__constant__ float d_zero = 0.0f;

__global__ void triangular_mm_kernel_warp_optimized(const float* __restrict__ A,
                                                    const float* __restrict__ B,
                                                    float* __restrict__ C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < d_N && col <= row) {
        float sum = 0.f;
        for (int k = col; k <= row; ++k) {
            sum += __ldg(&A[row * d_N + k]) * __ldg(&B[k * d_N + col]);
        }

        // Use warp-level reduction to sum the results
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }

        // Write the result from the first thread in the warp
        if (threadIdx.x % warpSize == 0) {
            C[row * d_N + col] = sum;
        }
    } else if (row < d_N && col < d_N) {
        // Write zeros to upper triangular part
        C[row * d_N + col] = d_zero;
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

    // Set L1 cache preference to prefer L1 over shared memory
    cudaFuncSetCacheConfig(triangular_mm_kernel_warp_optimized, cudaFuncCachePreferL1);

    triangular_mm_kernel_warp_optimized<<<numBlocks, threadsPerBlock>>>(
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