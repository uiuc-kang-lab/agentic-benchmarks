#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    // Use 1D thread indexing for simpler mapping
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Convert linear index to row/col
    const int row = tid / N;
    const int col = tid % N;

    // Shared memory for caching
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    if (row < N && col < N) {
        if (row < col) {
            // Upper triangle - just set to zero
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            
            // Process the matrix in tiles
            for (int t = 0; t < N; t += 32) {
                // Load tile into shared memory
                if (threadIdx.x < 32) {
                    const int r = row;
                    const int c = t + threadIdx.x;
                    As[threadIdx.x][threadIdx.x] = (c < N && r < N) ? A[r * N + c] : 0.0f;
                    Bs[threadIdx.x][threadIdx.x] = (c < N && col < N) ? B[c * N + col] : 0.0f;
                }
                __syncthreads();

                // Compute partial sum for this tile
                #pragma unroll 8
                for (int k = max(col, t); k < min(row + 1, t + 32); ++k) {
                    sum += As[threadIdx.x][k - t] * Bs[k - t][threadIdx.x];
                }
                __syncthreads();
            }
            
            C[row * N + col] = sum;
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

    // Use 1D grid and block configuration
    const int threadsPerBlock = 256;  // Optimal thread count for H100
    const int numBlocks = (N * N + threadsPerBlock - 1) / threadsPerBlock;

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