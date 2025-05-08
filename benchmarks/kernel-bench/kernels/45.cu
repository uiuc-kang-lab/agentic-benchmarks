#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define BLOCK_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

__global__ void matmul_optimized_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N) {
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;
    const int num_tiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Early exit if this thread is outside matrix bounds
    const bool valid_thread = (row < N && col < N);
    float C_value = 0.0f;

    for (int m = 0; m < num_tiles; ++m) {
        // Calculate source indices
        const int src_row = row;
        const int src_col = m * BLOCK_SIZE + tx;
        const int src_row_B = m * BLOCK_SIZE + ty;
        const int src_col_B = col;

        // Load tiles into shared memory with bounds checking
        As[ty][tx] = (src_row < N && src_col < N) ? A[src_row * N + src_col] : 0.0f;
        Bs[ty][tx] = (src_row_B < N && src_col_B < N) ? B[src_row_B * N + src_col_B] : 0.0f;

        __syncthreads();

        // Compute partial product - no branches in main computation
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            C_value += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    // Write the result - single branch at the end
    if (valid_thread) {
        C[row * N + col] = C_value;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int64_t N = A.size(0);

    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    matmul_optimized_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_data, B_data, C_data, N);

    // Check for kernel launch errors
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized matrix multiplication kernel (CUDA)");
}