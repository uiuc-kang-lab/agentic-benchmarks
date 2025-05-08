#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>

#ifndef TILE_DIM
#define TILE_DIM 16  // Experiment with 8, 16, or 32 to tune block size (i.e., 64, 256, or 1024 threads per block)
#endif

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Templated matrix multiplication kernel using shared memory tiling
template <int TILE_DIM>
__global__ void matmul_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C,
                                int M, int N, int K) {
    // Allocate shared memory for tiles of A and B
    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    float sum = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int A_col = t * TILE_DIM + threadIdx.x;
        int B_row = t * TILE_DIM + threadIdx.y;

        // Load tile from A into shared memory
        if (row < M && A_col < K) {
            As[threadIdx.y][threadIdx.x] = A[row * K + A_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile from B into shared memory
        if (B_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[B_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();

        // Unrolled computation within the tile
        #pragma unroll
        for (int k = 0; k < TILE_DIM; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Write the result to global memory
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function called from PyTorch
torch::Tensor matmul_cuda(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    // Configure grid and block dimensions based on the tunable TILE_DIM
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    // Launch the templated kernel. To experiment with different block sizes, recompile with different TILE_DIM values
    matmul_kernel<TILE_DIM><<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, N, K
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_cuda, "Tuned Matrix Multiplication (CUDA) with templated block size");
}
