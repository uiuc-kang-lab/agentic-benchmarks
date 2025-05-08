#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// CUDA kernel optimized for memory coalescing
__global__ void coalesced_matmul_kernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int N) {
    __shared__ float shared_A[TILE_SIZE][TILE_SIZE];
    __shared__ float shared_B[TILE_SIZE][TILE_SIZE];

    // Compute global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load tile from A with coalesced access
        int a_col = m * TILE_SIZE + threadIdx.x;
        if (row < N && a_col < N)
            shared_A[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            shared_A[threadIdx.y][threadIdx.x] = 0.0f;

        // Load tile from B with coalesced access
        int b_row = m * TILE_SIZE + threadIdx.y;
        if (b_row < N && col < N)
            shared_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            shared_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the two tiles together
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += shared_A[threadIdx.y][k] * shared_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the result back to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// C++ interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    coalesced_matmul_kernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced Matrix Multiplication (CUDA)");
}
