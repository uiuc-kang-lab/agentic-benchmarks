#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Use a tile size that is a multiple of 4 to help with 128-bit (16 byte) alignment
#define TILE_SIZE 16

// CUDA kernel for matrix multiplication using __ldg() for read-only global memory accesses
__global__ void matmul_kernel(const float* __restrict__ A,
                              const float* __restrict__ B,
                              float* __restrict__ C,
                              int N) {
    // Shared memory tiles. Using float arrays; assuming that global memory pointers
    // for A and B are 16-byte aligned (i.e. aligned to 128-bit boundaries) for optimal load efficiency
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Compute row and column indices for the output matrix
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float value = 0.0f;

    // Number of tiles along the K dimension
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = 0; i < numTiles; i++) {
        // Compute global indices for the tile loads
        int aCol = i * TILE_SIZE + tx;
        int bRow = i * TILE_SIZE + ty;

        // Load element of A using __ldg() which hints a read-only access and can help
        // with caching. We assume that the underlying data is 16-byte aligned if possible.
        s_A[ty][tx] = (row < N && aCol < N) ? __ldg(&A[row * N + aCol]) : 0.0f;

        // Similarly, load element of B
        s_B[ty][tx] = (bRow < N && col < N) ? __ldg(&B[bRow * N + col]) : 0.0f;

        __syncthreads();

        // Compute partial product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            value += s_A[ty][k] * s_B[k][tx];
        }

        __syncthreads();
    }

    // Write the result. Global stores are not marked with __ldg() since C is written to
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// C++ interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    // Ensure the inputs are CUDA tensors, are 2D, square, and of the same size
    TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Tensor B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Only 2D matrices are supported");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "Matrices must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrices must be of the same size");

    int N = A.size(0);
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Configure thread block and grid dimensions
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication using __ldg() and 128-bit aligned accesses (CUDA)");
}
