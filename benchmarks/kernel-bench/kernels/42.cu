#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Macros for input checks
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// Templated tiled kernel. TILE_SIZE is a compile-time constant that will allow us to experiment with various block dimensions.
template <int TILE_SIZE>
__global__ void matmul_tiled_kernel(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C,
                                    int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float C_value = 0.0f;

    // Loop over the tiles
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        // Load elements into shared memory
        if (row < N && m * TILE_SIZE + tx < N)
            As[ty][tx] = A[row * N + m * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if (col < N && m * TILE_SIZE + ty < N)
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        // Unroll the inner loop to help the compiler optimize multiplication-add sequences
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            C_value += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = C_value;
}

// Forward function with an additional parameter 'tile_size' to experiment with different block configurations.
// Allowed tile sizes correspond to the dimension of the tile (resulting in tile_size * tile_size threads per block).
// For example: 8 (64 threads/block), 16 (256 threads/block), or 32 (1024 threads/block).

torch::Tensor forward(torch::Tensor A, torch::Tensor B, int tile_size = 16) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threadsPerBlock(tile_size, tile_size);
    dim3 blocksPerGrid((N + tile_size - 1) / tile_size, (N + tile_size - 1) / tile_size);

    // Choose the kernel instantiation based on the provided tile_size.
    switch(tile_size) {
        case 8:
            matmul_tiled_kernel<8><<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);
            break;
        case 16:
            matmul_tiled_kernel<16><<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);
            break;
        case 32:
            matmul_tiled_kernel<32><<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);
            break;
        default:
            TORCH_CHECK(false, "Unsupported tile size. Supported sizes: 8, 16, 32");
    }

    // Check for any kernel launch errors
    C10_CUDA_CHECK(cudaGetLastError());
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix multiplication kernel with block size experimentation (CUDA)");
}
