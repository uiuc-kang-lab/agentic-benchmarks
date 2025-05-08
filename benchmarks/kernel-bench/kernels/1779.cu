#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * blockDim.y + ty;
    const int col = blockIdx.x * blockDim.x + tx;

    float sum = 0.0f;

    if (row < N && col < N && row >= col) {
        const int numTiles = (min(row + 1, N) - col + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        for (int t = 0; t < numTiles; ++t) {
            const int tileStart = col + t * BLOCK_SIZE;
            
            if (row < N && (tileStart + tx) <= row) {
                As[ty][tx] = A[row * N + (tileStart + tx)];
            } else {
                As[ty][tx] = 0.0f;
            }
            
            if ((tileStart + ty) < N && col < N) {
                Bs[ty][tx] = B[(tileStart + ty) * N + col];
            } else {
                Bs[ty][tx] = 0.0f;
            }
            
            __syncthreads();

            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                if (tileStart + k <= row) {
                    sum += As[ty][k] * Bs[k][tx];
                }
            }
            
            __syncthreads();
        }
        
        C[row * N + col] = sum;
    } else if (row < N && col < N) {
        C[row * N + col] = 0.0f;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    auto C = torch::empty_like(A);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    triangular_mm_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}