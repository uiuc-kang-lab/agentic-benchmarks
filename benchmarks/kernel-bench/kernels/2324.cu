#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size
#define TILE_SIZE 32

// Optimized kernel using tiling with shared memory, __ldg for read-only caching, and restrict qualifiers
__global__ void matmul_tiled_ldg_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    // Calculate row and column index for this thread
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float c_val = 0.0f;

    // Loop over tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        int k_offset = t * TILE_SIZE;

        // Load A tile from global memory using __ldg for read-only data caching
        if (row < M && (k_offset + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + k_offset + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B tile from global memory using __ldg (B is accessed as if transposed)
        if (col < N && (k_offset + threadIdx.y) < K) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[col * K + k_offset + threadIdx.y]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Unroll inner loop to improve performance
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            c_val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write the computed value to the output if within the matrix boundary
    if (row < M && col < N) {
        C[row * N + col] = c_val;
    }
}

// Host interface using PyTorch C++ extension
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());
    
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
    dim3 block(TILE_SIZE, TILE_SIZE);

    // Launch kernel
    matmul_tiled_ldg_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix multiplication (A * B^T) utilizing tiling, shared memory, __ldg caching, and restrict qualifiers");
}
