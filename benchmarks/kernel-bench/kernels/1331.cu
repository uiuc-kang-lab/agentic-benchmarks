#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void uniform_diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x;
    if (row < N) {
        float a_val = A[row];
        int offset = row * M;
        int tid = threadIdx.x;
        int stride = blockDim.x;
        const int unroll_factor = 4;
        int step = stride * unroll_factor;
        int main_end = (M / step) * step;

        // Unrolled loop for the main computation
        #pragma unroll
        for (int col = tid; col < main_end; col += step) {
            #pragma unroll
            for (int k = 0; k < unroll_factor; ++k) {
                int j = col + k * stride;
                C[offset + j] = a_val * B[offset + j];
            }
        }

        // Process any remaining columns
        for (int col = main_end + tid; col < M; col += stride) {
            C[offset + col] = a_val * B[offset + col];
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

    // Launch one block per row, using 256 threads per block
    int threads = 256;
    dim3 grid(N);
    uniform_diag_matmul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Uniform control flow diagonal matrix multiplication");
}