#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define MAX_MATRIX_DIM 8192

__constant__ int d_N;  // Matrix dimension in constant memory
__constant__ int d_num_tiles;  // Number of tiles needed for the computation

// CUDA kernel for matrix multiplication
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    
    float value = 0;

    for (int i = 0; i < d_num_tiles; ++i) {
        if (row < d_N && i * TILE_SIZE + tx < d_N)
            s_A[ty][tx] = A[row * d_N + i * TILE_SIZE + tx];
        else
            s_A[ty][tx] = 0.0f;

        if (col < d_N && i * TILE_SIZE + ty < d_N)
            s_B[ty][tx] = B[(i * TILE_SIZE + ty) * d_N + col];
        else
            s_B[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k)
            value = fmaf(s_A[ty][k], s_B[k][tx], value);

        __syncthreads();
    }

    if (row < d_N && col < d_N)
        C[row * d_N + col] = value;
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
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds maximum supported size");

    int N = A.size(0);
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Copy constants to device constant memory
    cudaMemcpyToSymbol(d_N, &N, sizeof(int));
    cudaMemcpyToSymbol(d_num_tiles, &num_tiles, sizeof(int));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Launch the CUDA kernel
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication with Constant Memory (CUDA)");
}