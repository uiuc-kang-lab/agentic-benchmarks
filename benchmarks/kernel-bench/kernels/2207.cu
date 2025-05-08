#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define WARP_SIZE 32
#define BLOCK_SIZE_M 8
#define BLOCK_SIZE_N 8
#define BLOCK_SIZE_K 32

__global__ void matmul2DKernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              const int K, const int M, const int N) {
    // Block index
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    
    // Thread index
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    // Index of the first sub-matrix of A processed by the block
    const int block_row = bx * BLOCK_SIZE_M;
    const int block_col = by * BLOCK_SIZE_N;

    // Shared memory declaration
    __shared__ float As[BLOCK_SIZE_K][BLOCK_SIZE_M];
    __shared__ float Bs[BLOCK_SIZE_K][BLOCK_SIZE_N];

    // Each thread computes one element in the block sub-matrix
    float Csub = 0.0f;

    // Loop over all sub-matrices of A and B required to compute the block sub-matrix
    for (int k = 0; k < K; k += BLOCK_SIZE_K) {
        // Load the matrices from global memory to shared memory
        // Each thread loads one element of each matrix
        
        // Collaborative loading of A into shared memory
        if ((block_row + tx) < M && (k + ty) < K) {
            As[ty][tx] = __ldg(&A[(k + ty) * M + (block_row + tx)]);
        } else {
            As[ty][tx] = 0.0f;
        }
        
        // Collaborative loading of B into shared memory
        if ((k + tx) < K && (block_col + ty) < N) {
            Bs[tx][ty] = __ldg(&B[(k + tx) * N + (block_col + ty)]);
        } else {
            Bs[tx][ty] = 0.0f;
        }
        
        __syncthreads();

        // Multiply the two matrices together
        #pragma unroll 8
        for (int kk = 0; kk < BLOCK_SIZE_K; ++kk) {
            Csub += As[kk][tx] * Bs[kk][ty];
        }
        
        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    const int c_row = block_row + tx;
    const int c_col = block_col + ty;
    if (c_row < M && c_col < N) {
        C[c_row * N + c_col] = Csub;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Configure kernel launch parameters
    dim3 threadsPerBlock(BLOCK_SIZE_M, BLOCK_SIZE_N);
    dim3 numBlocks(
        (M + BLOCK_SIZE_M - 1) / BLOCK_SIZE_M,
        (N + BLOCK_SIZE_N - 1) / BLOCK_SIZE_N
    );

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    matmul2DKernel<<<numBlocks, threadsPerBlock>>>(
        A_ptr, B_ptr, C_ptr, K, M, N
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized 2D indexed matrix multiplication (CUDA)");
}