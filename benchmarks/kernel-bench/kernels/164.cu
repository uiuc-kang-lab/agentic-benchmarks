#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 32

#define CHECK_INPUT(x) \
    TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor"); \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous"); \
    TORCH_CHECK(x.scalar_type() == torch::kFloat, #x " must be a float tensor");

__global__ void optimizedMatrixMultiplyKernel(const float* __restrict__ A,
                                               const float* __restrict__ B,
                                               float* __restrict__ C,
                                               int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < numTiles; t++) {
        if (row < M && t * TILE_SIZE + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + t * TILE_SIZE + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < K) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[(t * TILE_SIZE + threadIdx.y) * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    optimizedMatrixMultiplyKernel<<<blocks, threads>>>(A.data_ptr<float>(),
                                                       B.data_ptr<float>(),
                                                       C.data_ptr<float>(),
                                                       M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Matrix Multiplication (CUDA)");
}