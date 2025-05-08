#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_SIZE_X 32
#define BLOCK_SIZE_Y 16

__global__ void block_size_512_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    const int N) {
    int row = blockIdx.y * BLOCK_SIZE_Y + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE_X + threadIdx.x;
    
    if (row < N && col < N && row <= col) {
        float sum = 0.0f;
        
        // Process elements in chunks to maximize instruction-level parallelism
        #pragma unroll 4
        for (int k = row; k <= col; k++) {
            sum += __ldg(&A[row * N + k]) * __ldg(&B[k * N + col]);
        }
        
        C[row * N + col] = sum;
    }
}

torch::Tensor block_size_512_matmul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);
    
    // Configure larger block size for potentially better occupancy
    dim3 threadsPerBlock(BLOCK_SIZE_X, BLOCK_SIZE_Y);  // 32x16 = 512 threads per block
    dim3 numBlocks((N + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X,
                   (N + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
    
    block_size_512_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &block_size_512_matmul, "Block size 512 optimized upper triangular matrix multiplication");
}