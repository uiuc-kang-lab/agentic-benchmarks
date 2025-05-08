#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Constant memory declaration for matrix B
__constant__ float B_const[16384];  // Holds up to 128x128 matrix

__global__ void triangular_mm_const_b(const float* __restrict__ A,
                                      float* __restrict__ C,
                                      int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        if (row < col) {
            C[row * N + col] = 0.f;
        } else {
            float sum = 0.f;
            // Access B from constant memory instead of global
            for (int k = col; k <= row; ++k) {
                sum += A[row * N + k] * B_const[k * N + col];
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
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    // Copy B matrix to constant memory
    size_t b_size = N * N * sizeof(float);
    TORCH_CHECK(b_size <= sizeof(B_const), "B matrix too large for constant memory");
    cudaMemcpyToSymbol(B_const, B.data_ptr<float>(), b_size);

    // Kernel configuration
    const int TILE = 16;
    dim3 blocks((N + TILE-1)/TILE, (N + TILE-1)/TILE);
    dim3 threads(TILE, TILE);

    triangular_mm_const_b<<<blocks, threads>>>(A.data_ptr<float>(),
                                             C.data_ptr<float>(),
                                             N);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular MM with B in constant memory");
}
