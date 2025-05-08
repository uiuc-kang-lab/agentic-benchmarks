#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel using shared memory, read-only cache, and aligned accesses
template<const int BLOCK_SIZE = 32>
__global__ void triangular_mm_kernel_optimized(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             const int N) {
    // Shared memory for tile-based computation
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles
    for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        // Load tile into shared memory
        const int tileStart = tile * BLOCK_SIZE;
        if (row < N && (tileStart + threadIdx.x) < N) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * N + tileStart + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((tileStart + threadIdx.y) < N && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[(tileStart + threadIdx.y) * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();

        // Compute partial sum for this tile
        if (row < N && col < N && row >= col) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                if ((tileStart + k) >= col && (tileStart + k) <= row) {
                    sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                }
            }
        }
        
        __syncthreads();
    }

    // Write result
    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.0f;
        } else {
            C[row * N + col] = sum;
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Inputs must be 2D tensors");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be same size");
    
    const int N = A.size(0);
    auto C = torch::empty_like(A);
    
    constexpr int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    triangular_mm_kernel_optimized<BLOCK_SIZE><<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}