#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 16

template <typename scalar_t>
__device__ __forceinline__ void load_tile_A(
    scalar_t* tile,
    const scalar_t* __restrict__ A,
    const int row,
    const int K,
    const int tile_idx,
    const int threadIdx_x,
    const int threadIdx_y) {
    
    int aCol = tile_idx * TILE_DIM + threadIdx_x;
    if (aCol < K) {
        tile[threadIdx_y * TILE_DIM + threadIdx_x] = A[row * K + aCol];
    } else {
        tile[threadIdx_y * TILE_DIM + threadIdx_x] = 0;
    }
}

template <typename scalar_t>
__device__ __forceinline__ void load_tile_B(
    scalar_t* tile,
    const scalar_t* __restrict__ B,
    const int col,
    const int K,
    const int L,
    const int tile_idx,
    const int threadIdx_x,
    const int threadIdx_y) {
    
    int bRow = tile_idx * TILE_DIM + threadIdx_y;
    if (bRow < K && col < L) {
        tile[threadIdx_y * TILE_DIM + threadIdx_x] = B[bRow * L + col];
    } else {
        tile[threadIdx_y * TILE_DIM + threadIdx_x] = 0;
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_tile_product(
    const scalar_t* tile_A,
    const scalar_t* tile_B,
    const int threadIdx_x,
    const int threadIdx_y) {
    
    scalar_t sum = 0;
    #pragma unroll
    for (int k = 0; k < TILE_DIM; k++) {
        sum += tile_A[threadIdx_y * TILE_DIM + k] * 
               tile_B[k * TILE_DIM + threadIdx_x];
    }
    return sum;
}

template <typename scalar_t>
__global__ void modular_tensor_matrix_multiplication_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ output,
    const int num_rows,
    const int K,
    const int L) {

    __shared__ scalar_t shared_A[TILE_DIM * TILE_DIM];
    __shared__ scalar_t shared_B[TILE_DIM * TILE_DIM];

    const int row = blockIdx.y * TILE_DIM + threadIdx.y;
    const int col = blockIdx.x * TILE_DIM + threadIdx.x;
    
    scalar_t sum = 0;

    const int num_tiles = (K + TILE_DIM - 1) / TILE_DIM;
    
    for (int t = 0; t < num_tiles; t++) {
        if (row < num_rows) {
            load_tile_A<scalar_t>(shared_A, A, row, K, t,
                                threadIdx.x, threadIdx.y);
        }
        
        load_tile_B<scalar_t>(shared_B, B, col, K, L, t,
                             threadIdx.x, threadIdx.y);
        
        __syncthreads();
        
        sum += compute_tile_product<scalar_t>(shared_A, shared_B,
                                            threadIdx.x, threadIdx.y);
        
        __syncthreads();
    }

    if (row < num_rows && col < L) {
        output[row * L + col] = sum;
    }
}

void module_fn_cuda_forward(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor output) {

    const int N = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int L = B.size(1);
    const int num_rows = N * M;

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((L + TILE_DIM - 1) / TILE_DIM,
              (num_rows + TILE_DIM - 1) / TILE_DIM);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "modular_tensor_matrix_multiplication", ([&] {
        modular_tensor_matrix_multiplication_kernel<scalar_t><<<grid, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            num_rows, K, L);
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in module_fn_cuda_forward: %s\n", cudaGetErrorString(err));
    }
}

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor module_fn_forward(
    torch::Tensor A,
    torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    auto N = A.size(0);
    auto M = A.size(1);
    auto L = B.size(1);

    auto output = torch::zeros({N, M, L}, A.options());
    module_fn_cuda_forward(A, B, output);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn_forward, "Modular tensor-matrix multiplication (CUDA)");
}