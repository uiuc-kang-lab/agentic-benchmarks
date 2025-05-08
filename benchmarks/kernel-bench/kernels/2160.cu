#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size and maximum dimension for constant memory usage
#define TILE_SIZE 32
#define MAX_DIM 128
#define MAX_SIZE (MAX_DIM * MAX_DIM)

// Declare constant memory for matrix B
__constant__ float constB[MAX_SIZE];

// CUDA kernel for lower triangular matrix multiplication using constant memory for B
__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                       float* __restrict__ C,
                                       const int N) {
    __shared__ float shA[TILE_SIZE][TILE_SIZE];
    __shared__ float shB[TILE_SIZE][TILE_SIZE];

    // Compute row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (row >= N || col >= N) return;
    // For lower triangular matrices, elements where row < col are zero
    if (row < col) {
        C[row * N + col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    
    // Determine tile range based on column and row indices
    int t_start = col / TILE_SIZE;
    int t_end = row / TILE_SIZE;

    #pragma unroll
    for (int t = t_start; t <= t_end; t++) {
        // Load a tile of matrix A from global memory into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        if (a_col < N && a_col <= row)
            shA[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            shA[threadIdx.y][threadIdx.x] = 0.0f;

        // Load a tile of matrix B from constant memory into shared memory
        int b_row = t * TILE_SIZE + threadIdx.y;
        if (b_row < N && b_row >= col)
            shB[threadIdx.y][threadIdx.x] = constB[b_row * N + col];
        else
            shB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Determine the effective range for k within this tile
        int k_begin = t * TILE_SIZE;
        if (k_begin < col) k_begin = col;
        int k_end = (t + 1) * TILE_SIZE;
        if (k_end > row + 1) k_end = row + 1;

        #pragma unroll
        for (int k = k_begin; k < k_end; k++) {
            int local_k = k - t * TILE_SIZE;
            sum += shA[threadIdx.y][local_k] * shB[local_k][threadIdx.x];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

// C++ interface exposed to PyTorch
at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D tensors");
    TORCH_CHECK(A.size(0) == A.size(1) && B.size(0) == B.size(1), "A and B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be the same size");

    const int N = A.size(0);
    // Ensure the matrix dimension fits within the constant memory limits
    TORCH_CHECK(N <= MAX_DIM, "Matrix dimension exceeds constant memory limit (", MAX_DIM, ")");

    auto C = torch::empty_like(A);

    // Copy matrix B into constant memory. B is assumed to be in row-major order.
    cudaError_t err = cudaMemcpyToSymbol(constB, B.data_ptr<float>(), N * N * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    TORCH_CHECK(err == cudaSuccess, "Copying B to constant memory failed: ", cudaGetErrorString(err));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel without passing B (since B is in constant memory now)
    triangular_mm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        C.data_ptr<float>(),
        N
    );

    err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular matrix multiplication (CUDA) using constant memory for B");
}
