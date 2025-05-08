#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

__global__ void matmul_stride_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int M, int K, int N) {
    // Stride loop implementation
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < M; row += blockDim.y * gridDim.y) {
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < N; col += blockDim.x * gridDim.x) {
            float sum = 0.0f;
            #pragma unroll
            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    const int BLOCK_DIM_X = 32;
    const int BLOCK_DIM_Y = 4;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_stride_kernel<<<grid, block>>>(A.data_ptr<float>(), 
                                        B.data_ptr<float>(), 
                                        C.data_ptr<float>(),
                                        M, K, N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride loop matrix multiplication (CUDA)");
}