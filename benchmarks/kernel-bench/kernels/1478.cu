#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Experiment with a larger block size of 32 to improve occupancy and reduce latency
#define BLOCK_SIZE 64

// Optimized CUDA kernel for symmetric matrix multiplication
__global__ void matmul_opt_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    __shared__ float s_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float value = 0.0f;

    // Loop over tiles
    for (int i = 0; i < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        // Load a tile of A into shared memory
        if (row < N && i * BLOCK_SIZE + tx < N)
            s_A[ty][tx] = A[row * N + i * BLOCK_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;

        // Load a tile of B into shared memory
        if (col < N && i * BLOCK_SIZE + ty < N)
            s_B[ty][tx] = B[(i * BLOCK_SIZE + ty) * N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            value += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = value;
}

// C++ interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Validate inputs: A and B must be square CUDA tensors of float type
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Launch configuration with BLOCK_SIZE x BLOCK_SIZE threads
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the optimized CUDA kernel
    matmul_opt_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Matrix Multiplication for Symmetric Matrices (CUDA)");
}
