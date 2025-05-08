#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 16
#define VECTOR_SIZE 4  // For vectorized loads

__global__ void bmm_tiled_unrolled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int batch_size,
    const int M,
    const int K,
    const int N
) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row = by * TILE + ty;
    const int col = bx * TILE + tx;

    // Pre-calculate batch offsets
    const float* A_batch = A + bz * M * K;
    const int batch_offset_b = bz * K * N;
    
    float sum = 0.0f;

    // Main computation loop
    #pragma unroll 4
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE + tx < K) {
            As[ty][tx] = A[batch_offset_a + row * K + t * TILE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }

        if (t * TILE + ty < K && col < N) {
            Bs[ty][tx] = B[batch_offset_b + (t * TILE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Manual unroll of the multiplication loop
        #pragma unroll
        for (int k = 0; k < TILE; k += 4) {
            sum += As[ty][k] * Bs[k][tx];
            sum += As[ty][k+1] * Bs[k+1][tx];
            sum += As[ty][k+2] * Bs[k+2][tx];
            sum += As[ty][k+3] * Bs[k+3][tx];
        }

        __syncthreads();
    }

    // Write result
    if (row < M && col < N) {
        C[bz * M * N + row * N + col] = sum;
    }
}

torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");

    const int batch_size = A.size(0);
    const int M = A.size(1);
    const int K = A.size(2);
    const int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch_size);

    bmm_tiled_unrolled_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Unrolled batched matrix multiplication (CUDA)");
}