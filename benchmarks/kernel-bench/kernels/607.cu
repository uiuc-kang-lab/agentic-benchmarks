#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define MAX_B_ELEM_FLOAT 16384  // Maximum number of float elements for matrix B in constant memory (64KB total)
#define MAX_B_ELEM_DOUBLE 8192  // Maximum number of double elements for matrix B in constant memory (64KB total)

// Declare constant memory for matrix B for float and double types
__constant__ float constB_float[MAX_B_ELEM_FLOAT];
__constant__ double constB_double[MAX_B_ELEM_DOUBLE];

// Helper function to fetch an element from constant memory for matrix B
// This function is specialized for float and double

template <typename scalar_t>
__device__ inline scalar_t getB(int row, int col, int N);

template <>
__device__ inline float getB<float>(int row, int col, int N) {
    return constB_float[row * N + col];
}

template <>
__device__ inline double getB<double>(int row, int col, int N) {
    return constB_double[row * N + col];
}

// CUDA kernel that performs matrix multiplication using shared memory tiling
// Matrix B is stored in constant memory to accelerate read-only access.

template <typename scalar_t>
__global__ void matmul_constB_kernel(const scalar_t* __restrict__ A,
                                       scalar_t* __restrict__ C,
                                       int M, int K, int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t value = 0;

    int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_WIDTH + threadIdx.x;
        int bRow = t * TILE_WIDTH + threadIdx.y;
        
        // Load a tile of A from global memory using __ldg for read-only caching
        if (row < M && aCol < K)
            sA[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + aCol]);
        else
            sA[threadIdx.y][threadIdx.x] = 0;

        // Load a tile of B from constant memory
        if (bRow < K && col < N)
            sB[threadIdx.y][threadIdx.x] = getB<scalar_t>(bRow, col, N);
        else
            sB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i++) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = value;
}

// Host function: copies matrix B to constant memory (if it fits within limits) and launches the kernel

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    // Copy matrix B into constant memory if it fits within hardware limits
    auto dtype = A.scalar_type();
    if (dtype == at::kFloat) {
        TORCH_CHECK(B.numel() <= MAX_B_ELEM_FLOAT, "Matrix B is too large for constant memory (float)");
        cudaMemcpyToSymbol(constB_float, B.data_ptr<float>(), B.numel() * sizeof(float));
    } else if (dtype == at::kDouble) {
        TORCH_CHECK(B.numel() <= MAX_B_ELEM_DOUBLE, "Matrix B is too large for constant memory (double)");
        cudaMemcpyToSymbol(constB_double, B.data_ptr<double>(), B.numel() * sizeof(double));
    } else {
        TORCH_CHECK(false, "Unsupported data type");
    }

    auto C = torch::empty({M, N}, A.options());
    
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 numBlocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_constB_kernel", ([&] {
        matmul_constB_kernel<scalar_t><<<numBlocks, threadsPerBlock>>>(
            A.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    }));

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication using constant memory for matrix B (CUDA)");
}
