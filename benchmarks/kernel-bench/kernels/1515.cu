#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

// CUDA kernel to perform matrix multiplication using symmetric properties
__global__ void matmul_kernel_symmetric(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float value = 0;

    for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; ++i) {
        // Avoiding divergence by loading within bounds
        float aElem = (row < N && i * TILE_SIZE + tx < N) ? A[row * N + i * TILE_SIZE + tx] : 0.0f;
        float bElem = (col < N && i * TILE_SIZE + ty < N) ? B[(i * TILE_SIZE + ty) * N + col] : 0.0f;
        s_A[ty][tx] = aElem;
        s_B[ty][tx] = bElem;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            value = fmaf(s_A[ty][k], s_B[k][tx], value);

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = value;
}

// C++ interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Check that A and B are float tensors, 2D, square, on CUDA
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    int N = A.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Launch the CUDA kernel
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel_symmetric<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication Symmetric (CUDA)");
}