#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel remains unchanged, but execution is optimized using streams
__global__ void diag_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int64_t N,
    const int64_t M
) {
    int row = blockIdx.x;  // one block per row
    if (row < N) {
        float a_val = A[row];
        int main_end = (M / blockDim.x) * blockDim.x;
        for (int j = threadIdx.x; j < main_end; j += blockDim.x) {
            int idx = row * M + j;
            C[idx] = a_val * B[idx];
        }
        for (int j = main_end + threadIdx.x; j < M; j += blockDim.x) {
            int idx = row * M + j;
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

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int threads = (M > 256) ? 256 : (((M + 31) / 32) * 32);
    dim3 grid(N);
    diag_matmul_kernel<<<grid, threads, 0, stream>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        N,
        M
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Diagonal matrix multiplication optimized with streams");
}