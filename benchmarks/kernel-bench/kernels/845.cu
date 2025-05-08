#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32
#define TILE_K 32

__global__ void vectorized_matmul_kernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int M, int K, int N) {
    __shared__ float As[BLOCK_SIZE][TILE_K + 1];
    __shared__ float Bs[BLOCK_SIZE][TILE_K + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * BLOCK_SIZE + ty;
    int col = blockIdx.x * BLOCK_SIZE + tx;

    float4 tmp;
    float sum[2][2] = {{0.0f}};

    for (int t = 0; t < (K + TILE_K - 1) / TILE_K; ++t) {
        // Vectorized load A tile with coalesced access
        int a_col = t * TILE_K + tx;
        if (row < M && a_col < K) {
            tmp = reinterpret_cast<const float4*>(A + row * K + a_col)[0];
            As[ty][tx] = tmp.x;
            As[ty][tx + 8] = tmp.y;
            As[ty][tx + 16] = tmp.z;
            As[ty][tx + 24] = tmp.w;
        } else {
            As[ty][tx] = 0;
            As[ty][tx + 8] = 0;
            As[ty][tx + 16] = 0;
            As[ty][tx + 24] = 0;
        }

        // Transposed load B tile with coalesced access
        int b_row = t * TILE_K + tx;
        if (b_row < K && col < N) {
            tmp = reinterpret_cast<const float4*>(B + b_row * N + col)[0];
            Bs[tx][ty] = tmp.x;
            Bs[tx][ty + 8] = tmp.y;
            Bs[tx][ty + 16] = tmp.z;
            Bs[tx][ty + 24] = tmp.w;
        } else {
            Bs[tx][ty] = 0;
            Bs[tx][ty + 8] = 0;
            Bs[tx][ty + 16] = 0;
            Bs[tx][ty + 24] = 0;
        }

        __syncthreads();

        // Compute with register blocking
        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            float a0 = As[ty][k];
            float a1 = As[ty + 8][k];
            float b0 = Bs[k][tx];
            float b1 = Bs[k][tx + 8];

            sum[0][0] += a0 * b0;
            sum[0][1] += a0 * b1;
            sum[1][0] += a1 * b0;
            sum[1][1] += a1 * b1;
        }
        __syncthreads();
    }

    // Store results with boundary checks
    if (row < M && col < N) C[row * N + col] = sum[0][0];
    if (row < M && (col + 8) < N) C[row * N + col + 8] = sum[0][1];
    if ((row + 8) < M && col < N) C[(row + 8) * N + col] = sum[1][0];
    if ((row + 8) < M && (col + 8) < N) C[(row + 8) * N + col + 8] = sum[1][1];
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_CUDA(A); CHECK_CONTIGUOUS(A);
    CHECK_CUDA(B); CHECK_CONTIGUOUS(B);

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    dim3 block(BLOCK_SIZE, 8);
    dim3 grid((N + BLOCK_SIZE-1)/BLOCK_SIZE, (M + BLOCK_SIZE-1)/BLOCK_SIZE);

    vectorized_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(),
                                             B.data_ptr<float>(),
                                             C.data_ptr<float>(),
                                             M, K, N);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Vectorized coalesced matrix multiplication (CUDA)");
}