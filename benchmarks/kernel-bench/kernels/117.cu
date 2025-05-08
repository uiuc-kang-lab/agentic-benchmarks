#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define TILE_SIZE 32
#define THREAD_TILE_X 4
#define THREAD_TILE_Y 4
#define BLOCK_ROWS (TILE_SIZE / THREAD_TILE_Y)
#define BLOCK_COLS (TILE_SIZE / THREAD_TILE_X)

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

static cublasHandle_t handle = nullptr;

__global__ void matmul_kernel_unrolled(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      const int M, const int N, const int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;

    float accum[THREAD_TILE_Y][THREAD_TILE_X] = {0.0f};
    
    const int row_base = by + ty * THREAD_TILE_Y;
    const int col_base = bx + tx * THREAD_TILE_X;

    #pragma unroll 1
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        #pragma unroll
        for (int i = 0; i < THREAD_TILE_Y; ++i) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_X; ++j) {
                const int row = by + ty * THREAD_TILE_Y + i;
                const int col = tile * TILE_SIZE + tx * THREAD_TILE_X + j;
                if (row < M && col < K) {
                    As[ty * THREAD_TILE_Y + i][tx * THREAD_TILE_X + j] = A[row * K + col];
                } else {
                    As[ty * THREAD_TILE_Y + i][tx * THREAD_TILE_X + j] = 0.0f;
                }
            }
        }

        #pragma unroll
        for (int i = 0; i < THREAD_TILE_Y; ++i) {
            #pragma unroll
            for (int j = 0; j < THREAD_TILE_X; ++j) {
                const int row = tile * TILE_SIZE + ty * THREAD_TILE_Y + i;
                const int col = bx + tx * THREAD_TILE_X + j;
                if (row < K && col < N) {
                    Bs[ty * THREAD_TILE_Y + i][tx * THREAD_TILE_X + j] = B[row * N + col];
                } else {
                    Bs[ty * THREAD_TILE_Y + i][tx * THREAD_TILE_X + j] = 0.0f;
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a_reg[THREAD_TILE_Y];
            float b_reg[THREAD_TILE_X];

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_Y; ++i) {
                a_reg[i] = As[ty * THREAD_TILE_Y + i][k];
            }

            #pragma unroll
            for (int j = 0; j < THREAD_TILE_X; ++j) {
                b_reg[j] = Bs[k][tx * THREAD_TILE_X + j];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_TILE_Y; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_TILE_X; ++j) {
                    accum[i][j] += a_reg[i] * b_reg[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < THREAD_TILE_Y; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_TILE_X; ++j) {
            const int row = row_base + i;
            const int col = col_base + j;
            if (row < M && col < N) {
                C[row * N + col] = accum[i][j];
            }
        }
    }
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    if (M <= 128 && N <= 128 && K <= 128) {
        dim3 threads(BLOCK_COLS, BLOCK_ROWS);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);
        
        matmul_kernel_unrolled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    } else {
        if (handle == nullptr) {
            cublasCreate(&handle);
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        
        cublasSgemm(handle,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int N = B.size(1);

    auto options = torch::TensorOptions()
                       .dtype(A.dtype())
                       .device(A.device())
                       .requires_grad(false);
    
    torch::Tensor C = torch::empty({M, N}, options);
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Unrolled hybrid matrix multiplication (CUDA)");
}