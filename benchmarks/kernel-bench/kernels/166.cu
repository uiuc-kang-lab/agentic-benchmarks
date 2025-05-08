#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>

#define TILE_SIZE 32
#define USE_CUSTOM_KERNEL_THRESHOLD (1 << 20)  // threshold on M*N*K

// Macros for input checking
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

// Custom tiled matrix multiplication kernel with manual unrolling and read-only cache usage
__global__ void matrixMultiplyKernel(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       const int M, const int N, const int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load A tile into shared memory using read-only cache
        int aCol = t * TILE_SIZE + threadIdx.x;
        if (row < M && aCol < K)
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + aCol]);
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        // Load B tile into shared memory
        int bRow = t * TILE_SIZE + threadIdx.y;
        if (bRow < K && col < N)
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[bRow * N + col]);
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Unrolled accumulation loop over the tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += 8) {
            sum += As[threadIdx.y][k]     * Bs[k][threadIdx.x]     +
                   As[threadIdx.y][k + 1] * Bs[k + 1][threadIdx.x] +
                   As[threadIdx.y][k + 2] * Bs[k + 2][threadIdx.x] +
                   As[threadIdx.y][k + 3] * Bs[k + 3][threadIdx.x] +
                   As[threadIdx.y][k + 4] * Bs[k + 4][threadIdx.x] +
                   As[threadIdx.y][k + 5] * Bs[k + 5][threadIdx.x] +
                   As[threadIdx.y][k + 6] * Bs[k + 6][threadIdx.x] +
                   As[threadIdx.y][k + 7] * Bs[k + 7][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Hybrid matrix multiply function that chooses between a custom kernel and CUBLAS based on problem size
void hybrid_matrix_multiply_cuda(const torch::Tensor &A, 
                                   const torch::Tensor &B, 
                                   torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    // Use custom kernel for small problem sizes
    if ((M * N * K) < USE_CUSTOM_KERNEL_THRESHOLD) {
        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                    (M + TILE_SIZE - 1) / TILE_SIZE);

        matrixMultiplyKernel<<<blocks, threads>>>(
            A.data_ptr<float>(),
            B.data_ptr<float>(),
            C.data_ptr<float>(),
            M, N, K
        );
        // Optional: check cuda error
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        }
    } else {
        // For larger matrices, use cuBLAS's highly optimized SGEMM
        cublasHandle_t handle;
        cublasCreate(&handle);

        // cuBLAS is column-major, but since our tensors are row-major we can swap A and B
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Note: the dimensions are adjusted to call cuBLAS sgemm correctly given row-major data.
        // We compute C = A * B, so we call: C^T = B^T * A^T
        // cublasSgemm expects:
        //    gemm(handle, transB, transA, N, M, K, ... , d_B, N, d_A, K, ..., d_C, N);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                    N, M, K, 
                    &alpha,
                    B.data_ptr<float>(), N,
                    A.data_ptr<float>(), K, 
                    &beta,
                    C.data_ptr<float>(), N);

        cublasDestroy(handle);
    }
}

// The forward function exposed to Python

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());
    hybrid_matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Hybrid Matrix Multiplication (CUDA) combining custom tiling and cuBLAS");
}
