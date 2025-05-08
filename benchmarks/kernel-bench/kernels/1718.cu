#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16
#define WARP_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float sum = 0.0f;
    
    if (row < N && col < N) {
        if (row >= col) {  // Only compute for lower triangular part
            // Loop over blocks
            for (int b = col; b <= row; b += BLOCK_SIZE) {
                int remaining = min(BLOCK_SIZE, row - b + 1);
                
                // Load data into shared memory
                if (b + tx < N && ty < remaining) {
                    As[ty][tx] = A[row * N + (b + tx)];
                    Bs[ty][tx] = B[(b + ty) * N + col];
                } else {
                    As[ty][tx] = 0.0f;
                    Bs[ty][tx] = 0.0f;
                }
                
                __syncthreads();
                
                // Compute partial dot product
                #pragma unroll
                for (int k = 0; k < remaining; k++) {
                    sum += As[ty][k] * Bs[k][tx];
                }
                
                __syncthreads();
            }
            
            // Warp-level reduction for final sum
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            // Write result
            if (threadIdx.x == 0) {
                C[row * N + col] = sum;
            }
        } else {
            C[row * N + col] = 0.0f;  // Upper triangular part is zero
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

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA)");
}