#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16

__global__ void strided_bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int total_elements = batch_size * M * N;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < total_elements) {
        const int b = idx / (M * N);
        const int remainder = idx % (M * N);
        const int m = remainder / N;
        const int n = remainder % N;

        float sum = 0.0f;
        
        for (int t = 0; t < K; t += TILE_SIZE) {
            const int tile_end = min(t + TILE_SIZE, K);
            const int tx = threadIdx.x % TILE_SIZE;
            const int ty = threadIdx.x / TILE_SIZE;

            // Load A tile
            if (ty < TILE_SIZE && m < M && t + tx < K) {
                As[ty][tx] = A[b * M * K + m * K + t + tx];
            } else {
                As[ty][tx] = 0.0f;
            }

            // Load B tile
            if (tx < TILE_SIZE && t + ty < K && n < N) {
                Bs[ty][tx] = B[b * K * N + (t + ty) * N + n];
            } else {
                Bs[ty][tx] = 0.0f;
            }

            __syncthreads();

            for (int k = 0; k < tile_end - t; ++k) {
                sum += As[ty][k] * Bs[k][tx];
            }
            
            __syncthreads();
        }

        if (m < M && n < N) {
            C[b * M * N + m * N + n] = sum;
        }
        
        idx += gridDim.x * blockDim.x;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
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

    const int threads_per_block = 256;
    int total_elements = batch_size * M * N;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    strided_bmm_kernel<<<blocks, threads_per_block>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Strided tiled batched matrix multiplication (CUDA)");
}
