#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define TILE_DIM 32

__global__ void matrix_multiply_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     const int M, const int N, const int K) {
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x * TILE_DIM;
    const int by = blockIdx.y * TILE_DIM;
    
    const int row = by + ty;
    const int col = bx + tx;
    
    const bool valid_thread = (row < M && col < N);
    float sum = 0.0f;
    
    #pragma unroll 4
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        const int tile_idx = t * TILE_DIM;
        
        const int a_row = row;
        const int a_col = min(tile_idx + tx, K-1);
        const int b_row = min(tile_idx + ty, K-1);
        const int b_col = min(bx + tx, N-1);
        
        As[ty][tx] = A[a_row * K + a_col];
        Bs[ty][tx] = B[b_row * N + b_col];
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        if (t < (K + TILE_DIM - 1) / TILE_DIM - 1) {
            __syncthreads();
        }
    }
    
    if (valid_thread) {
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

    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

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
    m.def("forward", &forward, "Optimized sync matrix multiplication (CUDA)");
}