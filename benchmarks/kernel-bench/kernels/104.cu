#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel for matrix multiplication using shared memory
__global__ void matrixMultiplyShared(float *A, float *B, float *C, int M, int N, int K) {
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Shared memory for A and B
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    // Each thread block computes one sub-matrix Csub of C
    float Cvalue = 0.0;

    // Loop over all the sub-matrices of A and B that are required to compute Csub
    for (int m = 0; m < (K + 15) / 16; ++m) {

        // Load the matrices from global memory to shared memory
        if (m * 16 + col < K && blockRow * 16 + row < M)
            As[row][col] = A[(blockRow * 16 + row) * K + m * 16 + col];
        else
            As[row][col] = 0.0;

        if (m * 16 + row < K && blockCol * 16 + col < N)
            Bs[row][col] = B[(m * 16 + row) * N + blockCol * 16 + col];
        else
            Bs[row][col] = 0.0;

        // Synchronize to make sure the sub-matrices are loaded
        __syncthreads();

        // Multiply the two matrices together
        for (int e = 0; e < 16; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding computation is done before loading new sub-matrices
        __syncthreads();
    }

    // Write the block sub-matrix to global memory
    if (blockRow * 16 + row < M && blockCol * 16 + col < N)
        C[(blockRow * 16 + row) * N + blockCol * 16 + col] = Cvalue;
}

void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    float *d_A = A.data_ptr<float>();
    float *d_B = B.data_ptr<float>();
    float *d_C = C.data_ptr<float>();

    // Define block size
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);

    // Launch kernel
    matrixMultiplyShared<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    torch::Tensor C = torch::zeros({M, N}, A.options());

    matrix_multiply_cuda(A, B, C);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication using shared memory (CUDA)");
}
