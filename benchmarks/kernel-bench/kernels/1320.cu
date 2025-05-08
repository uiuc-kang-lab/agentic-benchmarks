#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that employs 2D grid-stride loops to evenly distribute work over both rows and columns
__global__ void diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    // Iterate over rows using grid-stride loop in the y-direction
    for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < N; row += blockDim.y * gridDim.y) {
        // Load the diagonal element for this row once into a register
        float a_val = A[row];
        
        // Iterate over columns using grid-stride loop in the x-direction
        for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < M; col += blockDim.x * gridDim.x) {
            int idx = row * M + col;
            C[idx] = a_val * B[idx];
        }
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.dim() == 1, "A must be a 1D tensor");
    TORCH_CHECK(B.dim() == 2, "B must be a 2D tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Dimension mismatch: A.size(0) must match B.size(0)");

    A = A.contiguous();
    B = B.contiguous();

    int64_t N = A.size(0);
    int64_t M = B.size(1);

    auto C = torch::empty({N, M}, B.options());

    // Define block dimensions (16x16 is a good balance) and grid dimensions computed to cover the entire matrix
    dim3 block(16, 16);
    dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    diag_matmul_kernel<<<grid, block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication using 2D grid-stride loops");
}
