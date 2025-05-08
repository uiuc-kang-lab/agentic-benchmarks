#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This CUDA kernel implements tiled matrix multiplication with memory coalescing
// by ensuring that threads in a warp access consecutive memory locations during
// global loads and stores. 
__global__ void coalesced_tiled_matmul_kernel(const float* __restrict__ A,
                                              const float* __restrict__ B,
                                              float* __restrict__ C,
                                              int M, int N, int K) {
    // Compute global row and column indices for C
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Shared memory tiles for A and B
    __shared__ float sh_A[TILE_SIZE][TILE_SIZE];
    __shared__ float sh_B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    // Loop over tiles required to cover K dimension
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // Calculate the column index for A and row index for B in the current tile
        int A_col = tile * TILE_SIZE + threadIdx.x;  // For A: row is fixed, col varies
        int B_row = tile * TILE_SIZE + threadIdx.y;  // For B: row varies, col is fixed
        
        // Load tile from A with coalesced access: threads in a warp read contiguous elements
        if (row < M && A_col < K) {
            sh_A[threadIdx.y][threadIdx.x] = A[row * K + A_col];
        } else {
            sh_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B with coalesced access: consecutive threads access consecutive memory locations
        if (B_row < K && col < N) {
            sh_B[threadIdx.y][threadIdx.x] = B[B_row * N + col];
        } else {
            sh_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute partial dot product for the tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += sh_A[threadIdx.y][k] * sh_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write back the computed value to C ensuring coalesced global memory write
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// PyTorch forward interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    torch::Tensor C = torch::empty({M, N}, options);

    // Define execution configuration
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    coalesced_tiled_matmul_kernel<<<gridDim, blockDim>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced tiled matrix multiplication (CUDA)");
}
