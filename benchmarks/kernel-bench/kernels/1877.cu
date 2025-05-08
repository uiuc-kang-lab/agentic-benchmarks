#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   const int N) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over blocks
    for (int bk = 0; bk <= (row / BLOCK_SIZE); bk++) {
        // Collaborative loading of A and B into shared memory
        const int block_start = bk * BLOCK_SIZE;
        
        if (row < N && (block_start + threadIdx.x) < N) {
            s_A[threadIdx.y][threadIdx.x] = A[row * N + block_start + threadIdx.x];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((block_start + threadIdx.y) < N && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[(block_start + threadIdx.y) * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();

        // Compute partial dot product using shared memory
        if (row < N && col < N && row >= col) {
            const int k_start = max(block_start, col);
            const int k_end = min(block_start + BLOCK_SIZE, row + 1);
            
            #pragma unroll 8
            for (int k = k_start; k < k_end; k++) {
                sum += s_A[threadIdx.y][k - block_start] * 
                       s_B[k - block_start][threadIdx.x];
            }
        }
        
        __syncthreads();
    }

    // Write result
    if (row < N && col < N) {
        if (row >= col) {
            C[row * N + col] = sum;
        } else {
            C[row * N + col] = 0.0f;
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

    const int N = A.size(0);
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