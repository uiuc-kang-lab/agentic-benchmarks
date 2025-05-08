#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 16

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            
            // Process the matrix in BLOCK_SIZE x BLOCK_SIZE tiles
            for (int tile = col / BLOCK_SIZE; tile <= row / BLOCK_SIZE; ++tile) {
                // Load tile into shared memory
                if (tile * BLOCK_SIZE + threadIdx.x <= row && 
                    threadIdx.y <= row) {
                    As[threadIdx.y][threadIdx.x] = A[row * N + tile * BLOCK_SIZE + threadIdx.x];
                } else {
                    As[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                if (tile * BLOCK_SIZE + threadIdx.y >= col && 
                    tile * BLOCK_SIZE + threadIdx.x <= row) {
                    Bs[threadIdx.y][threadIdx.x] = B[(tile * BLOCK_SIZE + threadIdx.y) * N + col];
                } else {
                    Bs[threadIdx.y][threadIdx.x] = 0.0f;
                }
                
                __syncthreads();
                
                // Compute partial dot product for this tile
                #pragma unroll
                for (int k = 0; k < BLOCK_SIZE; ++k) {
                    if (tile * BLOCK_SIZE + k >= col && 
                        tile * BLOCK_SIZE + k <= row) {
                        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                    }
                }
                
                __syncthreads();
            }
            
            // Warp-level reduction for the final sum
            #pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            if (threadIdx.x == 0) {
                C[row * N + col] = sum;
            }
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