#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

__constant__ int const_N;
__constant__ int const_num_tiles;

__global__ void matmul_const_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float C_value = 0.0f;

    #pragma unroll 4
    for (int m = 0; m < const_num_tiles; ++m) {
        if (row < const_N && m * TILE_SIZE + tx < const_N)
            As[ty][tx] = __ldg(&A[row * const_N + m * TILE_SIZE + tx]);
        else
            As[ty][tx] = 0.0f;

        if (col < const_N && m * TILE_SIZE + ty < const_N)
            Bs[ty][tx] = __ldg(&B[(m * TILE_SIZE + ty) * const_N + col]);
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            C_value = __fmaf_rn(As[ty][k], Bs[k][tx], C_value);
        }

        __syncthreads();
    }

    if (row < const_N && col < const_N)
        C[row * const_N + col] = C_value;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int N = static_cast<int>(A.size(0));
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    cudaMemcpyToSymbol(const_N, &N, sizeof(int));
    cudaMemcpyToSymbol(const_num_tiles, &num_tiles, sizeof(int));

    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(num_tiles, num_tiles);

    matmul_const_kernel<<<blocks, threads>>>(A_data, B_data, C_data);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with constant memory optimization");
}