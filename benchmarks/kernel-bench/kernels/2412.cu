#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void matmul_thread_efficiency_kernel(const float* __restrict__ A,
                                                 const float* __restrict__ B,
                                                 float* __restrict__ C,
                                                 int M, int N, int K) {
    // Calculate the row and column index each warp/block will compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float c_value = 0.0f;
        // Each thread computes one element in the result matrix
        for (int k = 0; k < K; k += 4) {
            float4 a = *(float4*)&A[row * K + k];
            float4 b = *(float4*)&B[col * K + k];
            c_value += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            c_value += A[row * K + k] * B[col * K + k];
        }
        C[row * N + col] = c_value;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dim() == 2, "A must be 2D");
    TORCH_CHECK(B.dim() == 2, "B must be 2D");
    TORCH_CHECK(A.size(1) == B.size(1), "A and B must have the same K dimension");
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be on CUDA");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "Inputs must be contiguous");

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    auto C = torch::empty({M, N}, A.options());

    // Optimize the grid dimensions for better memory access patterns
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    matmul_thread_efficiency_kernel<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Thread/bock optimized matrix multiplication with transposed B (CUDA)");
}