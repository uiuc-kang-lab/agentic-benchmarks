#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define MAX_MATRIX_DIM 64  // Limited by constant memory size

__constant__ float A_const[MAX_MATRIX_DIM * MAX_MATRIX_DIM];
__constant__ float B_const[MAX_MATRIX_DIM * MAX_MATRIX_DIM];

__global__ void triangular_mm_kernel_const(
    float* __restrict__ C,
    const int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            for (int k = col; k <= row; ++k) {
                sum += A_const[row * N + k] * B_const[k * N + col];
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
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds constant memory limit");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Copy input matrices to constant memory
    cudaMemcpyToSymbol(A_const, A.data_ptr<float>(), N * N * sizeof(float));
    cudaMemcpyToSymbol(B_const, B.data_ptr<float>(), N * N * sizeof(float));

    const int threads = 16;
    dim3 threadsPerBlock(threads, threads);
    dim3 numBlocks((N + threads - 1) / threads, (N + threads - 1) / threads);

    triangular_mm_kernel_const<<<numBlocks, threadsPerBlock>>>(
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