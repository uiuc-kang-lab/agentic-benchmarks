#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for batched matrix multiplication: C = A * B
// Shapes: A (batch_size, M, K), B (batch_size, K, N), C (batch_size, M, N)
__global__ void bmm_kernel_optimized(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    extern __shared__ float shared[];
    float* As = shared;
    float* Bs = shared + blockDim.x * blockDim.y;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0.0f;
    for (int t = 0; t < (K - 1) / blockDim.x + 1; ++t) {
        if (row < M && t * blockDim.x + tx < K) {
            As[ty * blockDim.x + tx] = A[(blockIdx.z * M + row) * K + t * blockDim.x + tx];
        } else {
            As[ty * blockDim.x + tx] = 0.0f;
        }

        if (t * blockDim.x + ty < K && col < N) {
            Bs[ty * blockDim.x + tx] = B[(blockIdx.z * K + t * blockDim.x + ty) * N + col];
        } else {
            Bs[ty * blockDim.x + tx] = 0.0f;
        }

        __syncthreads();

        for (int k = 0; k < blockDim.x; ++k) {
            sum += As[ty * blockDim.x + k] * Bs[k * blockDim.x + tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[(blockIdx.z * M + row) * N + col] = sum;
    }
}

torch::Tensor forward_bmm_optimized(torch::Tensor A, torch::Tensor B) {
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

    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y, batch_size);
    size_t shared_mem_size = 2 * threads.x * threads.y * sizeof(float);

    bmm_kernel_optimized<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm_optimized, "Optimized Batched matrix multiplication (CUDA)");
}