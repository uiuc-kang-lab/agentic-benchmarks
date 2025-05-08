#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32
#define TILE_DIM (BLOCK_SIZE * 2)
#define CUBLAS_THRESHOLD 512

__global__ void warp_unified_control_flow_matmul_kernel(const float* __restrict__ A,
                                                        const float* __restrict__ B,
                                                        float* __restrict__ C,
                                                        int M, int K, int N) {
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float Cvalue[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int tile = 0; tile < (K + TILE_DIM - 1) / TILE_DIM; tile++) {
        int tStart = tile * TILE_DIM;

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int aRow = blockIdx.y * TILE_DIM + threadIdx.y + i * BLOCK_SIZE;
            int aCol = tStart + threadIdx.x;
            As[threadIdx.y + i * BLOCK_SIZE][threadIdx.x] = (aRow < M && aCol < K) ? A[aRow * K + aCol] : 0.0f;
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int bRow = tStart + threadIdx.y;
            int bCol = blockIdx.x * TILE_DIM + threadIdx.x + i * BLOCK_SIZE;
            Bs[threadIdx.y][threadIdx.x + i * BLOCK_SIZE] = (bRow < K && bCol < N) ? B[bRow * N + bCol] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            float a_val0 = As[threadIdx.y][k];
            float a_val1 = As[threadIdx.y + BLOCK_SIZE][k];
            float b_val0 = Bs[k][threadIdx.x];
            float b_val1 = Bs[k][threadIdx.x + BLOCK_SIZE];
            Cvalue[0] += a_val0 * b_val0;
            Cvalue[1] += a_val0 * b_val1;
            Cvalue[2] += a_val1 * b_val0;
            Cvalue[3] += a_val1 * b_val1;
        }

        __syncthreads();
    }

    if (row < M) {
        if (col < N) C[row * N + col] = Cvalue[0];
        if (col + BLOCK_SIZE < N) C[row * N + col + BLOCK_SIZE] = Cvalue[1];
    }
    if (row + BLOCK_SIZE < M) {
        if (col < N) C[(row + BLOCK_SIZE) * N + col] = Cvalue[2];
        if (col + BLOCK_SIZE < N) C[(row + BLOCK_SIZE) * N + col + BLOCK_SIZE] = Cvalue[3];
    }
}

torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    if (M >= CUBLAS_THRESHOLD && N >= CUBLAS_THRESHOLD && K >= CUBLAS_THRESHOLD) {
        cublasHandle_t handle;
        cublasCreate(&handle);
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                   N, M, K, &alpha,
                   B.data_ptr<float>(), N,
                   A.data_ptr<float>(), K,
                   &beta, C.data_ptr<float>(), N);
        cublasDestroy(handle);
    } else {
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);
        warp_unified_control_flow_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), 
                                                                B.data_ptr<float>(), 
                                                                C.data_ptr<float>(), 
                                                                M, K, N);
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Warp unified control flow matrix multiplication (CUDA)");
}