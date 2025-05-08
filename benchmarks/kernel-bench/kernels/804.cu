#include <torch/extension.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 128
#define BLOCK_DIM_Y 8

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int M, int K, int N) {
    __shared__ float As[BLOCK_DIM_Y][BLOCK_DIM_X];
    __shared__ float Bs[BLOCK_DIM_X][BLOCK_DIM_Y];

    int row = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM_X + threadIdx.x;
    
    float sum = 0.0f;

    for (int t = 0; t < (K + BLOCK_DIM_X - 1)/BLOCK_DIM_X; ++t) {
        // Coalesced A tile load
        int a_col = t * BLOCK_DIM_X + threadIdx.x;
        if (row < M && a_col < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Coalesced B tile load
        int b_row = t * BLOCK_DIM_X + threadIdx.y;
        if (b_row < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Vectorized accumulation
        #pragma unroll
        for (int k = 0; k < BLOCK_DIM_X; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CONTIGUOUS(A);
    CHECK_CUDA(B); CHECK_CONTIGUOUS(B);

    int M = A.size(0), K = A.size(1), N = B.size(1);
    torch::Tensor C = torch::zeros({M, N}, A.options());

    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((N + BLOCK_DIM_X - 1)/BLOCK_DIM_X, (M + BLOCK_DIM_Y - 1)/BLOCK_DIM_Y);

    matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tuned block size matmul (CUDA)");
}