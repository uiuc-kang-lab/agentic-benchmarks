#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32
#define THREAD_TILE 2
#define MAX_MATRIX_DIM 8192

// Constant memory for small kernels
__constant__ float d_A[MAX_MATRIX_DIM * MAX_MATRIX_DIM];
__constant__ float d_B[MAX_MATRIX_DIM * MAX_MATRIX_DIM];

// Load matrices into constant memory before kernel execution
template <typename scalar_t>
void copyToConstantMemory(scalar_t* A, scalar_t* B, int N) {
    cudaMemcpyToSymbol(d_A, A, N * N * sizeof(scalar_t));
    cudaMemcpyToSymbol(d_B, B, N * N * sizeof(scalar_t));
}

// Optimized kernel utilizing constant memory
__global__ void matmul_kernel_const(float* __restrict__ C) {
    int blockRow = blockIdx.y * BLOCK_SIZE;
    int blockCol = blockIdx.x * BLOCK_SIZE;

    int ty = threadIdx.y;
    int tx = threadIdx.x;
    int row = blockRow + ty * THREAD_TILE;
    int col = blockCol + tx * THREAD_TILE;

    float regC00 = 0.0f, regC01 = 0.0f, regC10 = 0.0f, regC11 = 0.0f;

    __shared__ float s_B[BLOCK_SIZE][BLOCK_SIZE];

    for (int t = 0; t < !(N + BLOCK_SIZE - 1) / BLOCK_SIZE; t++) {
        // Load using constant memory for A
        int aRow0 = row;
        int aRow1 = row + 1;
        int aCol0 = t * BLOCK_SIZE + tx * THREAD_TILE;
        int aCol1 = aCol0 + 1;

        float a00 = (aRow0 < N && aCol0 < N) ? d_A[aRow0 * N + aCol0] : 0.0f;
        float a01 = (aRow0 < N && aCol1 < N) ? d_A[aRow0 * N + aCol1] : 0.0f;
        float a10 = (aRow1 < N && aCol0 < N) ? d_A[aRow1 * N + aCol0] : 0.0f;
        float a11 = (aRow1 < N && aCol1 < N) ? d_A[aRow1 * N + aCol1] : 0.0f;

        int bRow0 = t * BLOCK_SIZE + ty * THREAD_TILE;
        int bRow1 = bRow0 + 1;
        int bCol0 = col;
        int bCol1 = col + 1;

        s_B[ty * THREAD_TILE + 0][tx * THREAD_TILE + 0] = (bRow0 < N && bCol0 < N) ? d_B[bRow0 * N + bCol0] : 0.0f;
        s_B[ty * THREAD_TILE + 0][tx * THREAD_TILE + 1] = (bRow0 < N && bCol1 < N) ? d_B[bRow0 * N + bCol1] : 0.0f;
        s_B[ty * THREAD_TILE + 1][tx * THREAD_TILE + 0] = (bRow1 < N && bCol0 < N) ? d_B[bRow1 * N + bCol0] : 0.0f;
        s_B[ty * THREAD_TILE + 1][tx * THREAD_TILE + 1] = (bRow1 < N && bCol1 < N) ? d_B[bRow1 * N + bCol1] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++) {
            float b0 = s_B[k][tx * THREAD_TILE + 0];
            float b1 = s_B[k][tx * THREAD_TILE + 1];
            regC00 += a00 * b0;
            regC01 += a00 * b1;
            regC10 += a10 * b0;
            regC11 += a10 * b1;
            regC00 += a01 * s_B[k + 1][tx * THREAD_TILE + 0];
            regC01 += a01 * s_B[k + 1][tx * THREAD_TILE + 1];
            regC10 += a11 * s_B[k + 1][tx * THREAD_TILE + 0];
            regC11 += a11 * s_B[k + 1][tx * THREAD_TILE + 1];
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = regC00;
    if (row < N && (col + 1) < N)
        C[row * N + col + 1] = regC01;
    if ((row + 1) < N && col < N)
        C[(row + 1) * N + col] = regC10;
    if ((row + 1) < N && (col + 1) < N)
        C[(row + 1) * N + col + 1] = regC11;
}

// C++ interface exposed with Pybind11
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Both A and B must be 2D matrices");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must have the same dimensions");
    TORCH_CHECK(A.size(0) <= MAX_MATRIX_DIM, "Matrix dimension exceeds supported maximum size");

    int N = A.size(0);

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    // Copy A and B to constant memory
    copyToConstantMemory(A.data_ptr<float>(), B.data_ptr<float>(), N);

    dim3 threads(BLOCK_SIZE / THREAD_TILE, BLOCK_SIZE / THREAD_TILE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_kernel_const<<<blocks, threads>>>(C.data_ptr<float>());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication utilizing constant memory (CUDA)");
}