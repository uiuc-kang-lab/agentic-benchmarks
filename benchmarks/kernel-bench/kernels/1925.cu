#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__constant__ float const_A[1024 * 1024];
__constant__ float const_B[1024 * 1024];

__global__ void constant_memory_mm_kernel(const float* __restrict__ C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by + ty;
    int col = bx + tx;
    
    float sum = 0.0f;
    
    if (row >= col && row < N && col < N) {
        for (int tile = col / TILE_SIZE; tile <= row / TILE_SIZE; ++tile) {
            As[ty][tx] = const_A[row * N + (tile * TILE_SIZE + tx)];
            Bs[ty][tx] = const_B[(tile * TILE_SIZE + ty) * N + col];
            
            __syncthreads();
            
            for (int k = 0; k < TILE_SIZE; ++k) {
                if ((tile * TILE_SIZE + k) >= col && (tile * TILE_SIZE + k) <= row) {
                    sum += As[ty][k] * Bs[k][tx];
                }
            }
            
            __syncthreads();
        }
        
        if (row < N && col < N) {
            C[row * N + col] = sum;
        }
    } else if (row < N && col < N) {
        C[row * N + col] = 0.0f;
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

    // Copy matrices to constant memory
    cudaMemcpyToSymbol(const_A, A.data_ptr<float>(), A.numel() * sizeof(float));
    cudaMemcpyToSymbol(const_B, B.data_ptr<float>(), B.numel() * sizeof(float));

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    constant_memory_mm_kernel<<<numBlocks, threadsPerBlock>>>(
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Constant memory triangular matrix multiplication (CUDA)");
}
