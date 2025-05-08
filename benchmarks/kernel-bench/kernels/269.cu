#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Tile size for shared memory tiling
#define TILE 16

__device__ void load_tile_to_shared_memory(const float* __restrict__ matrix, float tile[][TILE], int row, int col, int dim0, int dim1) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int index = row * dim1 + col;

    if (row < dim0 && col < dim1) {
        tile[ty][tx] = matrix[index];
    } else {
        tile[ty][tx] = 0.0f;
    }
}

// Kernel for batched matrix multiplication:
__global__ void bmm_kernel_modular(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    extern __shared__ float shared[];
    float (*As)[TILE] = (float (*)[TILE])shared;
    float (*Bs)[TILE] = (float (*)[TILE])(shared + TILE * TILE);

    // Index calculation for the batch, row, and column
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE + ty;
    int col = bx * TILE + tx;

    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;
        
        load_tile_to_shared_memory(A + bz * M * K, As, row, a_col, M, K);
        load_tile_to_shared_memory(B + bz * K * N, Bs, b_row, col, K, N);

        __syncthreads();

        for (int k = 0; k < TILE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[(bz * M + row) * N + col] = sum;
    }
}

torch::Tensor forward_bmm_modular(torch::Tensor A, torch::Tensor B) {
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

    size_t shared_mem_size = 2 * TILE * TILE * sizeof(float);

    bmm_kernel_modular<<<blocks, threads, shared_mem_size>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm_modular, "Modular Batched Matrix Multiplication (CUDA)");
}