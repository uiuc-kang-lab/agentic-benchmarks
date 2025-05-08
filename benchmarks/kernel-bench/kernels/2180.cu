#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_DIM 16

// Device function that computes the dot product over a tile loaded in shared memory.
// Each thread computes its partial sum using its thread indices tx and ty.
__device__ float tileMultiply(const float A_tile[TILE_DIM][TILE_DIM],
                                const float B_tile[TILE_DIM][TILE_DIM],
                                int tx, int ty) {
    float partial_sum = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_DIM; ++k) {
        partial_sum += A_tile[k][tx] * B_tile[k][ty];
    }
    return partial_sum;
}

// Kernel implementing C = A.T * B using a tiled approach.
// A has shape (K, M) and B has shape (K, N). Thus, C (result) has shape (M, N) and is computed as:
//   C(i, j) = sum_{k=0}^{K-1} A[k * M + i] * B[k * N + j]
// The kernel is modularized using the tileMultiply device function for the inner product over a tile.
__global__ void modularTiledKernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int K,
                                   int M,
                                   int N) {
    // Compute the output element indices.
    int i = blockIdx.x * TILE_DIM + threadIdx.x; // Corresponds to row in C (and column in A)
    int j = blockIdx.y * TILE_DIM + threadIdx.y; // Corresponds to column in C (and column in B)

    float sum = 0.0f;

    // Declare shared memory for tiles of A and B.
    __shared__ float A_tile[TILE_DIM][TILE_DIM];
    __shared__ float B_tile[TILE_DIM][TILE_DIM];

    // Loop over the K dimension in tiles of size TILE_DIM.
    for (int t = 0; t < K; t += TILE_DIM) {
        // Load a tile of A. For A, each tile element is A[(t + row) * M + i] where row = threadIdx.y.
        if ((t + threadIdx.y) < K && i < M) {
            A_tile[threadIdx.y][threadIdx.x] = A[(t + threadIdx.y) * M + i];
        } else {
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Load a tile of B. For B, each tile element is B[(t + row) * N + j] where row = threadIdx.y.
        if ((t + threadIdx.y) < K && j < N) {
            B_tile[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + j];
        } else {
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Compute the partial product for this tile using the modularized device function.
        sum += tileMultiply(A_tile, B_tile, threadIdx.x, threadIdx.y);

        __syncthreads();
    }

    // Write the computed sum to the output matrix C, if within bounds.
    if (i < M && j < N) {
        C[i * N + j] = sum;
    }
}

// The forward function exposed via PyBind11. It sets up the grid and block dimensions and launches the kernel.
// A: Tensor of shape (K, M) [CUDA, float32]
// B: Tensor of shape (K, N) [CUDA, float32]
// Returns: Tensor C of shape (M, N) computed as A.T * B

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Validate inputs: they must be CUDA tensors and of type float32.
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    // Retrieve dimensions: A is (K, M) and B is (K, N), so C will be (M, N).
    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    // Allocate output tensor C on the same device and with the same data type.
    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    // Define grid and block dimensions based on the tile size.
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    // Launch the modular tiled kernel.
    modularTiledKernel<<<gridDim, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B with modular tiled kernel (CUDA)");
}
