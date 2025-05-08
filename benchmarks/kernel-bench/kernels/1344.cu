#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

// 2D kernel using block and grid indices to map threads to matrix elements
__global__ void diag_matmul_2d_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N,
    const int M
) {
    // Determine the row and column index for this thread
    int row = blockIdx.y * BLOCK_DIM_Y + threadIdx.y;
    int col = blockIdx.x * BLOCK_DIM_X + threadIdx.x;

    // Allocate shared memory for the diagonal values of A for the rows in this block
    __shared__ float sharedA[BLOCK_DIM_Y];

    // Each thread in the first column loads the corresponding A value if within bounds
    if (threadIdx.x == 0 && row < N) {
        sharedA[threadIdx.y] = A[row];
    }
    __syncthreads();

    // Compute only if within the bounds of the output matrix
    if (row < N && col < M) {
        int idx = row * M + col;
        float a_val = sharedA[threadIdx.y];
        C[idx] = a_val * B[idx];
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    // Ensure tensors are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int N = A.size(0);
    int M = B.size(1);
    auto C = torch::empty({N, M}, B.options());

    // Define a 2D block and grid configuration
    dim3 block(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 grid((M + BLOCK_DIM_X - 1) / BLOCK_DIM_X,
              (N + BLOCK_DIM_Y - 1) / BLOCK_DIM_Y);

    // Launch the kernel
    diag_matmul_2d_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N, M);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "2D diagonal matrix multiplication with shared memory optimization");
}
