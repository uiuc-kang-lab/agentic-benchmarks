#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel combining warp-level optimization with efficient memory access
__global__ void optimized_tril_mm_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        const int N) {
    // Use shared memory for tile-based computation
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];
    
    const int warpSize = 32;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;
    
    // Warp-aligned coordinates for early triangle check
    const int warpRow = row & ~(warpSize-1);
    const int warpCol = col & ~(warpSize-1);
    
    float sum = 0.0f;
    
    // Only process if warp potentially contains valid elements
    if (warpRow >= warpCol && row < N && col < N) {
        // Tile the computation
        for (int t = 0; t < N; t += 32) {
            // Collaborative loading of tiles into shared memory
            if (row < N && (t + tx) < N)
                As[ty][tx] = A[row * N + (t + tx)];
            else
                As[ty][tx] = 0.0f;
                
            if ((t + ty) < N && col < N)
                Bs[ty][tx] = B[(t + ty) * N + col];
            else
                Bs[ty][tx] = 0.0f;
                
            __syncthreads();
            
            // Compute on tiles
            if (row >= col) {
                #pragma unroll
                for (int k = 0; k < 32; ++k) {
                    if ((t + k) >= col && (t + k) <= row)
                        sum += As[ty][k] * Bs[k][tx];
                }
            }
            __syncthreads();
        }
        
        // Write result
        if (row >= col)
            C[row * N + col] = sum;
        else
            C[row * N + col] = 0.0f;
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

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + 31) / 32, (N + 31) / 32);

    optimized_tril_mm_kernel<<<numBlocks, threadsPerBlock>>>(
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
    m.def("forward", &forward, "Optimized triangular matrix multiplication (CUDA)");
}