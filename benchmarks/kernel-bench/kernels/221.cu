#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for batched matrix multiplication using warp-level primitives
// Shapes: A (batch_size, M, K), B (batch_size, K, N), C (batch_size, M, N)
__global__ void warp_bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    int batch = blockIdx.z;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float val = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = A[batch * M * K + row * K + k];
            float b = B[batch * K * N + k * N + col];
            val += a * b;
        }

        // Use warp-level reduction to sum up the values
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }

        // Write the result for the first thread in the warp
        if (threadIdx.x % warpSize == 0) {
            C[batch * M * N + row * N + col] = val;
        }
    }
}

torch::Tensor forward_warp_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y, batch_size);

    warp_bmm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_warp_bmm, "Batched matrix multiplication with warp-level primitives (CUDA)");
}