#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define BLOCK_SIZE 32
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void matrix_multiply_kernel(const float* A, const float* B, float* C, 
                                     const int M, const int N, const int K) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Stride loop - each thread handles multiple elements
    const int row_stride = gridDim.y * blockDim.y;
    const int col_stride = gridDim.x * blockDim.x;
    
    for (int i = row; i < M; i += row_stride) {
        for (int j = col; j < N; j += col_stride) {
            float sum = 0.0f;
            
            // Compute dot product with manual unrolling
            #pragma unroll 4
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            
            C[i * N + j] = sum;
        }
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // Calculate grid dimensions to ensure coverage of the entire matrix
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(
        (N + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (M + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    // Launch kernel with computed dimensions
    matrix_multiply_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Strided matrix multiplication (CUDA)");
}