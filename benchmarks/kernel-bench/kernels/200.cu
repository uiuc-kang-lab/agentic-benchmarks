#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define TILE_SIZE 16
// Maximum number of floats that can be stored in constant memory for matrix B
#define MAX_B_SIZE (1024 * 1024)

// Declare constant memory for matrix B
__constant__ float const_B[MAX_B_SIZE];

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Kernel that performs matrix multiplication using constant memory for B
__global__ void matrix_multiply_constB_kernel(const float* A, float* C, int M, int N, int K) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles in the K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int tiledA_col = t * TILE_SIZE + threadIdx.x;
        if (row < M && tiledA_col < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + tiledA_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int tiledB_row = t * TILE_SIZE + threadIdx.y;  // Row index for B
        if (tiledB_row < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = const_B[tiledB_row * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Launcher function that copies B to constant memory and launches the kernel
void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Ensure that matrix B fits in constant memory
    int b_elements = B.numel();
    TORCH_CHECK(b_elements <= MAX_B_SIZE, "Matrix B is too large to fit in constant memory (provided ", b_elements,
                " elements, maximum allowed is ", MAX_B_SIZE, ").");

    // Copy matrix B to constant memory
    cudaMemcpyToSymbol(const_B, B.data_ptr<float>(), b_elements * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    const float* d_A = A.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matrix_multiply_constB_kernel<<<gridSize, blockSize>>>(d_A, d_C, M, N, K);

    cudaDeviceSynchronize();
}

// Pybind11 interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    // Create output tensor
    torch::Tensor C = torch::zeros({M, N}, A.options());
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication using constant memory for B (CUDA)");
}
