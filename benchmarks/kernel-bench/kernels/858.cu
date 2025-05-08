#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32
#define TILE_DIM (BLOCK_SIZE * 2)
#define CUBLAS_THRESHOLD 512

__global__ void warp_primitive_matmul_kernel(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float Cvalue = 0.0f;

    for (int tile = 0; tile < (K + BLOCK_SIZE - 1) / BLOCK_SIZE; tile++) {
        float a_value = (row < M && tile * BLOCK_SIZE + threadIdx.x < K) ?
            A[row * K + tile * BLOCK_SIZE + threadIdx.x] : 0.0f;
        float b_value = (col < N && tile * BLOCK_SIZE + threadIdx.y < K) ?
            B[(tile * BLOCK_SIZE + threadIdx.y) * N + col] : 0.0f;

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; k++) {
            float a = __shfl_sync(0xffffffff, a_value, k);
            float b = __shfl_sync(0xffffffff, b_value, k);
            Cvalue += a * b;
        }
    }

    if (row < M && col < N) {
        C[row * N + col] = Cvalue;
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
        dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
        warp_primitive_matmul_kernel<<<grid, block>>>(A.data_ptr<float>(), 
                                                      B.data_ptr<float>(), 
                                                      C.data_ptr<float>(), 
                                                      M, K, N);
    }
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Warp primitive optimized matrix multiplication (CUDA)");
}