#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 32

__global__ void bmm_tiled_shared_memory_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
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

    float sum = 0.0f;
    float4 a_reg, b_reg;

    const int batch_offset_A = bz * M * K;
    const int batch_offset_B = bz * K * N;
    const int num_tiles = (K + TILE - 1) / TILE;

    int a_col = tx;
    int b_row = ty;
    
    // Main loop with prefetching
    for (int t = 0; t < num_tiles; t++) {
        // Load current tile
        if (row < M && (t * TILE + tx) < K) {
            As[ty][tx] = A[batch_offset_A + row * K + (t * TILE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }

        if ((t * TILE + ty) < K && col < N) {
            Bs[ty][tx] = B[batch_offset_B + (t * TILE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        // Compute on current tile
        #pragma unroll
        for (int i = 0; i < TILE; i++) {
            sum = __fmaf_rn(As[ty][i], Bs[i][tx], sum);
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[bz * M * N + row * N + col] = sum;
    }
}

torch::Tensor forward_bmm_shared_memory(torch::Tensor A, torch::Tensor B) {
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

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch_size);

    bmm_tiled_shared_memory_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm_shared_memory, "Batched matrix multiplication with shared memory optimization (CUDA)");
}