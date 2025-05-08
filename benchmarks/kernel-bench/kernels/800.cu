#include <torch/extension.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Tiling parameters
#define TILE_DIM 16

__global__ void matmul_tile_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int M, int K, int N) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    __shared__ float A_tile[TILE_DIM][TILE_DIM];
    __shared__ float B_tile[TILE_DIM][TILE_DIM];

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        if (row < M && t * TILE_DIM + threadIdx.x < K)
            A_tile[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_DIM + threadIdx.x];
        else
            A_tile[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && t * TILE_DIM + threadIdx.y < K)
            B_tile[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        else
            B_tile[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += A_tile[threadIdx.y][i] * B_tile[i][threadIdx.x];
        }

        __syncthreads();
    }
    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    const int BLOCK_DIM_X = 32;
    const int BLOCK_DIM_Y = 4;
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_stride_kernel<<<grid, block>>>(A.data_ptr<float>(), 
                                        B.data_ptr<float>(), 
                                        C.data_ptr<float>(),
                                        M, K, N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride loop matrix multiplication (CUDA)");
}