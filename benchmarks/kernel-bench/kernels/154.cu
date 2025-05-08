#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

// Define tile size of 32 to leverage shared memory effectively
#define TILE_SIZE 32

// Macro to ensure the input tensors are CUDA, contiguous, and of float type
#define CHECK_INPUT(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK(x.scalar_type() == torch::kFloat, #x " must be a float tensor");

// CUDA kernel that performs matrix multiplication using shared memory tiling
__global__ void matrixMultiplyKernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M, int N, int K) {
    // Allocate shared memory for tiles of A and B
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Calculate row and column index for the thread in the global matrix
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float value = 0.0f;
    // Loop over tiles of A and B coming from the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        // Load an element of A into shared memory if within bounds
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load an element of B into shared memory if within bounds
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute the product for the tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            value += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Store the result in C if within bounds
    if (row < M && col < N) {
        C[row * N + col] = value;
    }
}

// C++ function that sets up and calls the CUDA kernel
void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Define the thread block dimensions and grid dimensions
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    matrixMultiplyKernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(A.data_ptr<float>(),
                                                B.data_ptr<float>(),
                                                C.data_ptr<float>(),
                                                M, N, K);

    // Check for any CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

// Pybind11 interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create output tensor C with the same options as A
    torch::Tensor C = torch::zeros({M, N}, A.options());
    
    // Execute the CUDA matrix multiplication
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication (CUDA optimized with shared memory tiling)");
}
