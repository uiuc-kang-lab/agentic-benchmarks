#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int BLOCK_SIZE = 16;

__global__ void strided_triangular_mm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N) {

    // Each thread handles multiple elements using strided loops
    for(int row = blockIdx.y * blockDim.y + threadIdx.y; 
        row < N; 
        row += gridDim.y * blockDim.y) {
        
        for(int col = blockIdx.x * blockDim.x + threadIdx.x;
            col < N;
            col += gridDim.x * blockDim.x) {

            if(row < col) {
                C[row * N + col] = 0.0f;
            } else {
                float sum = 0.0f;
                for(int k = col; k <= row; ++k) {
                    sum += A[row * N + k] * B[k * N + col];
                }
                C[row * N + col] = sum;
            }
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / (4 * BLOCK_SIZE),
              (N + BLOCK_SIZE - 1) / (4 * BLOCK_SIZE));

    strided_triangular_mm_kernel<<<grid, block>>>(
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
    m.def("forward", &forward, "Strided triangular matmul (CUDA)");
}
