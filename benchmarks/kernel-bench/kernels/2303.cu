#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

const int TILE_SIZE = 32;

__device__ void load_tile(const float* matrix, float tile[TILE_SIZE][TILE_SIZE], int row, int col, int stride, int K) {
    int tx = threadIdx.x, ty = threadIdx.y;
    if (row < stride && (col + tx) < K) {
        tile[ty][tx] = matrix[row * K + col + tx];
    } else {
        tile[ty][tx] = 0.0;
    }
}

__device__ float compute_tile_product(float As[TILE_SIZE][TILE_SIZE], float Bs[TILE_SIZE][TILE_SIZE]) {
    float c_val = 0.0;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        c_val += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }
    return c_val;
}

__global__ void matmul_transposed_kernel_modular(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float c_val = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int k_offset = t * TILE_SIZE;

        load_tile(A, As, row, k_offset, M, K);
        load_tile(B, Bs, col, k_offset, N, K);

        __syncthreads();

        c_val += compute_tile_product(As, Bs);

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = c_val;
    }
}

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
    
    matmul_transposed_kernel_modular<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with transposed B modular (CUDA)");
}