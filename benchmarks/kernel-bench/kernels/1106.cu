#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32

// Device function to load a tile from matrix A into shared memory
template <typename scalar_t>
__device__ __forceinline__ void load_tile_A(const scalar_t* __restrict__ A,
                                             scalar_t tile_A[TILE_DIM][TILE_DIM],
                                             int row,
                                             int tile_col_offset,
                                             int K) {
    int col = tile_col_offset + threadIdx.x;
    tile_A[threadIdx.y][threadIdx.x] = (col < K) ? __ldg(&A[row * K + col]) : scalar_t(0);
}

// Device function to load a tile from matrix B into shared memory
template <typename scalar_t>
__device__ __forceinline__ void load_tile_B(const scalar_t* __restrict__ B,
                                             scalar_t tile_B[TILE_DIM][TILE_DIM],
                                             int tile_row_offset,
                                             int col,
                                             int L,
                                             int K) {
    int row = tile_row_offset + threadIdx.y;
    tile_B[threadIdx.y][threadIdx.x] = (row < K && col < L) ? __ldg(&B[row * L + col]) : scalar_t(0);
}

// Device function to compute dot product for a tile
template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_tile_dot(scalar_t tile_A[TILE_DIM][TILE_DIM],
                                                      scalar_t tile_B[TILE_DIM][TILE_DIM]) {
    scalar_t sum = 0;
    #pragma unroll
    for (int i = 0; i < TILE_DIM; i++) {
        sum += tile_A[threadIdx.y][i] * tile_B[i][threadIdx.x];
    }
    return sum;
}

// Modular tiled kernel for 3D tensor-matrix multiplication
// Multiplies a 3D tensor A of dimensions [N, M, K] with a matrix B of dimensions [K, L]
// Resulting in an output tensor of dimensions [N, M, L] by flattening the first two dimensions of A
// and using shared memory tiling to reduce global memory accesses.
template <typename scalar_t>
__global__ void modular_tiled_kernel(const scalar_t* __restrict__ A,
                                       const scalar_t* __restrict__ B,
                                       scalar_t* __restrict__ output,
                                       int N, int M, int K, int L) {
    // Flatten the first two dimensions of A: total rows = N * M
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    scalar_t sum = 0;
    
    __shared__ scalar_t tile_A[TILE_DIM][TILE_DIM];
    __shared__ scalar_t tile_B[TILE_DIM][TILE_DIM];
    
    int numTiles = (K + TILE_DIM - 1) / TILE_DIM;
    for (int t = 0; t < numTiles; t++) {
        int tile_offset = t * TILE_DIM;
        if (row < N * M) {
            load_tile_A<scalar_t>(A, tile_A, row, tile_offset, K);
        } else {
            tile_A[threadIdx.y][threadIdx.x] = scalar_t(0);
        }
        load_tile_B<scalar_t>(B, tile_B, tile_offset, col, L, K);
        __syncthreads();
        
        sum += compute_tile_dot<scalar_t>(tile_A, tile_B);
        __syncthreads();
    }
    
    if (row < N * M && col < L)
        output[row * L + col] = sum;
}

// CUDA forward function that launches the modular tiled kernel
void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {
    int N = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int L = B.size(1);
    
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM, ((N * M) + TILE_DIM - 1) / TILE_DIM);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "modular_tiled_kernel", ([&] {
        modular_tiled_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            N, M, K, L);
    }));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in modular_tiled_kernel: %s\n", cudaGetErrorString(err));
    }
}

// Macros for tensor input checks
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// C++ interface to launch the CUDA kernel using Pybind11
torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    
    int N = A.size(0);
    int M = A.size(1);
    int L = B.size(1);
    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Modular tiled tensor-matrix multiplication (CUDA)");
}
