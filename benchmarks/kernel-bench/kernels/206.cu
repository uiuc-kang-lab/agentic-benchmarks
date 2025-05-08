#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define BLOCK_SIZE 64

__global__ void matrix_multiply_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const int M, const int N, const int K) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;
    
    float sum = 0.0f;
    
    // Prefetch first tile
    float prefetch_a = (row < M && tx < K) ? __ldg(&A[row * K + tx]) : 0.0f;
    float prefetch_b = (ty < K && col < N) ? __ldg(&B[ty * N + col]) : 0.0f;

    for (int t = 0; t < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t) {
        // Load current tile
        if (row < M && (t * BLOCK_SIZE + tx) < K) {
            As[ty][tx] = prefetch_a;
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if ((t * BLOCK_SIZE + ty) < K && col < N) {
            Bs[ty][tx] = prefetch_b;
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        // Prefetch next tile
        if (t < ((K + BLOCK_SIZE - 1) / BLOCK_SIZE - 1)) {
            int next_idx_a = row * K + (t + 1) * BLOCK_SIZE + tx;
            int next_idx_b = ((t + 1) * BLOCK_SIZE + ty) * N + col;
            prefetch_a = (row < M && ((t + 1) * BLOCK_SIZE + tx) < K) ? __ldg(&A[next_idx_a]) : 0.0f;
            prefetch_b = (((t + 1) * BLOCK_SIZE + ty) < K && col < N) ? __ldg(&B[next_idx_b]) : 0.0f;
        }
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k += 8) {
            sum += As[ty][k] * Bs[k][tx];
            sum += As[ty][k+1] * Bs[k+1][tx];
            sum += As[ty][k+2] * Bs[k+2][tx];
            sum += As[ty][k+3] * Bs[k+3][tx];
            sum += As[ty][k+4] * Bs[k+4][tx];
            sum += As[ty][k+5] * Bs[k+5][tx];
            sum += As[ty][k+6] * Bs[k+6][tx];
            sum += As[ty][k+7] * Bs[k+7][tx];
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

    const float *d_A = A.data_ptr<float>();
    const float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matrix_multiply_kernel<<<grid, threads>>>(d_A, d_B, d_C, M, N, K);
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int N = B.size(1);

    torch::Tensor C = torch::empty({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tuned block size matrix multiplication (CUDA)");
}