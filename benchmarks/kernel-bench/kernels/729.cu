#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// Define tile size for the kernel and maximum constant memory utilization
#define TILE_WIDTH 16
#define MAX_CONST_ELEMENTS 16384  // Maximum number of floats (64KB limit if float is 4 bytes)

// Declare constant memory for matrix B
__constant__ float const_B[MAX_CONST_ELEMENTS];

// Kernel that reads matrix B from constant memory
__global__ void MatmulKernelConstB(const float* __restrict__ A, float* __restrict__ C, int M, int K, int N) {
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // A is in global memory, B is read from constant memory
            float a = A[row * K + k];
            float b = const_B[k * N + col];
            sum += a * b;
        }
        C[row * N + col] = sum;
    }
}

// The forward function selects the kernel based on whether B fits in constant memory
// It copies B to constant memory if it is small enough.

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // If B fits within constant memory, use the constant memory kernel
    if (B.numel() <= MAX_CONST_ELEMENTS) {
        // Copy B to constant memory
        cudaMemcpyToSymbol(const_B, B.data_ptr<float>(), B.numel() * sizeof(float));
        
        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

        MatmulKernelConstB<<<gridDim, blockDim>>>(A.data_ptr<float>(), C.data_ptr<float>(), M, K, N);
    } else {
        // Fallback to cuBLAS if B is too large to fit in constant memory
        static cublasHandle_t handle = nullptr;
        if (handle == nullptr) {
            cublasCreate(&handle);
        }
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha,
                    B.data_ptr<float>(), N,
                    A.data_ptr<float>(), K,
                    &beta, C.data_ptr<float>(), N);
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication with constant memory for B (CUDA)");
}
