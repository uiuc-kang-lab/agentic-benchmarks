#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define MAX_CONST_SIZE 16384 // 16384 floats = 64KB

__constant__ float B_const[MAX_CONST_SIZE];

__global__ void matmul_kernel_const(const float* __restrict__ A,
                                    float* __restrict__ C,
                                    const int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float value = 0.0f;

    for (int m = 0; m < (N + TILE_SIZE - 1) / TILE_SIZE; ++m) {
        int a_col = m * TILE_SIZE + tx;
        if (row < N && a_col < N)
            s_A[ty][tx] = A[row * N + a_col];
        else
            s_A[ty][tx] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            int b_row = m * TILE_SIZE + k;
            float b_val = (b_row < N && col < N) ? B_const[b_row * N + col] : 0.0f;
            value += s_A[ty][k] * b_val;
        }

        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = value;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    int N = A.size(0);
    TORCH_CHECK(N * N <= MAX_CONST_SIZE, "B is too large for constant memory");

    cudaMemcpyToSymbol(B_const, B.data_ptr<float>(), N * N * sizeof(float));

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel_const<<<blocks, threads>>>(A.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication with Constant Memory (CUDA)");
}