#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32     
#define TILE_K 32         

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static cublasHandle_t cublas_handle = nullptr;

__global__ void matmul_kernel_pipelined(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int M, int N, int K) {
    __shared__ float As[BLOCK_SIZE][TILE_K];
    __shared__ float Bs[TILE_K][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float Cvalue = 0.0f;

    for (int tileIdx = 0; tileIdx < (K + TILE_K - 1) / TILE_K; ++tileIdx) {
        if (row < M && tileIdx * TILE_K + threadIdx.x < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + tileIdx * TILE_K + threadIdx.x];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (tileIdx * TILE_K + threadIdx.y < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(tileIdx * TILE_K + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < TILE_K; ++k) {
            Cvalue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    float* d_A = A.data_ptr<float>();
    float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    const int blocks_X = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_Y = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim(blocks_X, blocks_Y);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    if (M <= 128 && N <= 128 && K <= 128) {
        matmul_kernel_pipelined<<<gridDim, blockDim, 0, stream1>>>(d_A, d_B, d_C, M, N, K);
    } else {
        if (cublas_handle == nullptr) {
            cublasCreate(&cublas_handle);
            cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
        }
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasSetStream(cublas_handle, stream2);  // Ensure cuBLAS operations happen in stream2
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    }

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

// PyTorch forward interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    int M = A.size(0);
    int N = B.size(1);
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device()).requires_grad(false);
    torch::Tensor C = torch::empty({M, N}, options);
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Pipelined hybrid matrix multiplication (CUDA): custom kernel and cuBLAS with streams");
}
