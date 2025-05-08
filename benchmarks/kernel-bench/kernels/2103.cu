#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Tiled kernel using shared memory to ensure coalesced accesses
// Computes C = tril(A * B) where A and B are lower triangular matrices.

__global__ void triangular_mm_tiled(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int N) {
    __shared__ float shA[TILE_SIZE][TILE_SIZE];
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // Loop over tiles in the k dimension
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;

        // Load a tile of A from global memory (coalesced read per row)
        if(row < N && aCol < N)
            shA[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        else
            shA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load a tile of B from global memory (coalesced read per row)
        if(bRow < N && col < N)
            shB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        else
            shB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial product on the tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += shA[threadIdx.y][k] * shB[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write result, enforcing the lower-triangular output
    if (row < N && col < N) {
        C[row * N + col] = (row >= col) ? sum : 0.0f;
    }
}

// C++ interface exposed to PyTorch
at::Tensor forward(const at::Tensor& A, const at::Tensor& B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_tiled<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled coalesced triangular matrix multiplication (CUDA)");
}
