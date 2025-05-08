#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <assert.h>

// Define tile size and maximum elements for constant memory storage for matrix B
#define TILE_SIZE 16
#define MAX_B_CONSTANT_ELEMENTS 16384

// Declare constant memory for matrix B for float and double types
__constant__ float B_const_f[MAX_B_CONSTANT_ELEMENTS];
__constant__ double B_const_d[MAX_B_CONSTANT_ELEMENTS];

// Helper function to return pointer to constant memory based on type
template <typename scalar_t>
__device__ inline const scalar_t* getBConst();

template <>
__device__ inline const float* getBConst<float>() {
    return B_const_f;
}

template <>
__device__ inline const double* getBConst<double>() {
    return B_const_d;
}

// CUDA kernel using shared memory tiling for A and constant memory for matrix B
// Computes C = A^T * B^T, with A of shape (K x M) and B of shape (N x K), yielding C of shape (M x N).
// Access pattern:
//   A[k, m] = A[k * M + m]
//   B[n, k] = B_const[n * K + k] (loaded from constant memory)

template <typename scalar_t>
__global__ void matmul_transpose_const_kernel(
    const scalar_t* __restrict__ A,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {

    // Compute global row and column indices for C
    int m = blockIdx.y * TILE_SIZE + threadIdx.y; // row index in C (and A's 2nd dimension)
    int n = blockIdx.x * TILE_SIZE + threadIdx.x; // column index in C (and B's 1st dimension)

    scalar_t sum = 0;

    // Allocate shared memory for a tile of A (from global memory)
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE];

    // Number of tiles along the K dimension
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        // Compute the global k index for loading element from A
        int k_index = t * TILE_SIZE + threadIdx.x;
        
        // Load element of A into shared memory (A is transposed: A[k, m] at A[k * M + m])
        if (m < M && k_index < K) {
            A_shared[threadIdx.y][threadIdx.x] = A[k_index * M + m];
        } else {
            A_shared[threadIdx.y][threadIdx.x] = 0;
        }

        // Synchronize to ensure the A tile is fully loaded
        __syncthreads();

        // Each thread loads a tile of B from constant memory into a register array
        // B is stored as (N x K): B[n, k] at B_const[n * K + k]
        scalar_t b_tile[TILE_SIZE];
        for (int i = 0; i < TILE_SIZE; i++) {
            int k_val = t * TILE_SIZE + i;
            if (n < N && k_val < K) {
                b_tile[i] = getBConst<scalar_t>()[n * K + k_val];
            } else {
                b_tile[i] = 0;
            }
        }
        
        // No need for synchronization here because b_tile is private to each thread
        
        // Compute the partial dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += A_shared[threadIdx.y][i] * b_tile[i];
        }

        // Synchronize to ensure that A_shared is not overwritten before use in next iteration
        __syncthreads();
    }

    // Write the computed value to global memory if within bounds
    if (m < M && n < N) {
        C[m * N + n] = sum;
    }
}

// PyTorch interface: copies matrix B to constant memory and launches the kernel

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    // A has dimensions (K x M) and B has dimensions (N x K), resulting in C of dimensions (M x N)
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);

    // Ensure that B fits in the available constant memory
    TORCH_CHECK(B.numel() <= MAX_B_CONSTANT_ELEMENTS, "Matrix B is too large to fit in constant memory");

    auto C = torch::empty({M, N}, A.options());

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_transpose_const_kernel", ([&] {
        size_t bytes = B.numel() * sizeof(scalar_t);
        // Copy contents of B into constant memory
        if (std::is_same<scalar_t, float>::value) {
            cudaMemcpyToSymbol(B_const_f, B.data_ptr<scalar_t>(), bytes, 0, cudaMemcpyDeviceToDevice);
        } else {
            cudaMemcpyToSymbol(B_const_d, B.data_ptr<scalar_t>(), bytes, 0, cudaMemcpyDeviceToDevice);
        }

        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        matmul_transpose_const_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transposed inputs using constant memory for matrix B (CUDA)");
}
