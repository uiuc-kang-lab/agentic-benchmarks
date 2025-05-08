#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Kernel using stride loops to cover workloads larger than the available threads
__global__ void stride_matmul_kernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int K, int N) {
    // Calculate the starting row and column indices for this thread
    int rowStart = blockIdx.y * blockDim.y + threadIdx.y;
    int colStart = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate the strides to cover the entire matrix
    int rowStride = gridDim.y * blockDim.y;
    int colStride = gridDim.x * blockDim.x;

    // Loop over rows and columns using stride loops
    for (int row = rowStart; row < M; row += rowStride) {
        for (int col = colStart; col < N; col += colStride) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// The forward function that wraps the kernel launch
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A);
    CHECK_CUDA(B);
    CHECK_CONTIGUOUS(A);
    CHECK_CONTIGUOUS(B);
    
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);
    
    torch::Tensor C = torch::zeros({M, N}, A.options());
    
    // Using a 2D grid-stride looping pattern to cover all elements
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    
    stride_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication using grid-stride loops (CUDA)");
}
