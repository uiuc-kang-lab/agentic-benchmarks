#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

// Define tile size and maximum constant memory elements for matrix B
#define TILE_SIZE 32
// Maximum allowed elements in constant memory for matrix B (supports up to 128x128 matrices)
#define MAX_CONST_ELEMENTS 16384

// Declare constant memory for matrix B
__constant__ float constB[MAX_CONST_ELEMENTS];

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// CUDA kernel using constant memory for matrix B
__global__ void matmul_constant_kernel(const float* __restrict__ A, float* __restrict__ C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;
    float C_val = 0.0f;

    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int m = 0; m < numTiles; ++m) {
        int tiledACol = m * TILE_SIZE + tx;
        if (row < N && tiledACol < N) {
            As[ty][tx] = A[row * N + tiledACol];
        } else {
            As[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            int tiledBRow = m * TILE_SIZE + k;
            float B_val = (tiledBRow < N && col < N) ? constB[tiledBRow * N + col] : 0.0f;
            C_val += As[ty][k] * B_val;
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = C_val;
}

// Forward function: copies matrix B into constant memory and launches the kernel
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int64_t N = A.size(0);
    TORCH_CHECK(N * N <= MAX_CONST_ELEMENTS, "Matrix size too large for constant memory optimization (max N = 128)");

    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    // Copy matrix B into constant memory
    cudaError_t err = cudaMemcpyToSymbol(constB, B_data, N * N * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    C10_CUDA_CHECK(err);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_constant_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel using constant memory for matrix B (CUDA)");
}
