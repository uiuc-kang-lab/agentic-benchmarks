#include <torch/extension.h>
#include <cuda_runtime.h>
#include <iostream>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel using grid-stride loops for flexible workload handling
__global__ void MatMulKernel(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K) {
    // Compute initial row and col indices for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop over rows with grid-stride
    for (int i = row; i < M; i += blockDim.y * gridDim.y) {
        // Loop over columns with grid-stride
        for (int j = col; j < N; j += blockDim.x * gridDim.x) {
            float sum = 0.0f;
            // Since K is small, use a simple loop over K
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Host function acting as a PyTorch binding
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    // Define block and grid dimensions
    const int BLOCK_DIM_X = 16;
    const int BLOCK_DIM_Y = 16;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((N + BLOCK_DIM_X - 1) / BLOCK_DIM_X, (M + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // Launch the kernel using grid-stride loops for full coverage
    MatMulKernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    cudaDeviceSynchronize();

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with grid-stride loops (CUDA)");
}
