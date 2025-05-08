#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define TILE_SIZE 32
#define MAX_B_SIZE 128  // Maximum supported matrix dimension for B constant memory usage

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

// Declare constant memory for matrix B
__constant__ float const_B[MAX_B_SIZE * MAX_B_SIZE];

__global__ void matmul_constB_kernel(const float* __restrict__ A, float* __restrict__ C, int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float C_value = 0.0f;

    // Loop over tiles
    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        int A_col = m * TILE_SIZE + tx;
        // Load A tile into shared memory
        if (row < N && A_col < N) {
            As[ty][tx] = A[row * N + A_col];
        } else {
            As[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute partial products using constant memory for B
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            int B_row = m * TILE_SIZE + k;
            float B_val = (B_row < N && col < N) ? const_B[B_row * N + col] : 0.0f;
            C_value += As[ty][k] * B_val;
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = C_value;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");

    int64_t N = A.size(0);
    TORCH_CHECK(N <= MAX_B_SIZE, "Matrix dimension ", N, " exceeds constant memory limit ", MAX_B_SIZE);

    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    // Copy matrix B into constant memory
    cudaMemcpyToSymbol(const_B, B_data, N * N * sizeof(float), 0, cudaMemcpyDeviceToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    matmul_constB_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, C_data, N);

    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiplication kernel with B in constant memory (CUDA)");
}
