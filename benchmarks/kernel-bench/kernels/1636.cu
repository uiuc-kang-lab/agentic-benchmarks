#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cuda_profiler_api.h>

#define TILE_DIM 16

// This kernel leverages shared memory tiling to reduce global memory latency. 
// Each block loads a tile of A and a tile of B into shared memory before computing partial dot products. 
// The valid range of k is checked to ensure only contributions from k in [row, col] are accumulated for the upper triangular matrix multiplication.

__global__ void upper_triangular_matmul_shared_kernel(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    // Loop over tiles along the k-dimension
    for (int tileStart = 0; tileStart < N; tileStart += TILE_DIM) {
        int aCol = tileStart + threadIdx.x;
        #if __CUDA_ARCH__ >= 800
        if (row < N && aCol < N) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" : : "r"(&As[threadIdx.y][threadIdx.x]), "l"(&A[row * N + aCol]), "n"(sizeof(float)));
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
#else
        if (row < N && aCol < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
#endif

        int bRow = tileStart + threadIdx.y;
        if (col < N && bRow < N) {
            Bs[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        // Use the shared memory tile to accumulate the partial sum
        for (int t = 0; t < TILE_DIM; ++t) {
            int k = tileStart + t;
            if (k < N && k >= row && k <= col) {
                sum += As[threadIdx.y][t] * Bs[t][threadIdx.x];
            }
        }
        __syncthreads();
    }

    // Write the result if within bounds and for upper triangular matrix
    if (row < N && col < N && row <= col) {
        C[row * N + col] = sum;
    }
}


torch::Tensor upper_triangular_matmul_shared(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros_like(A);

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    upper_triangular_matmul_shared_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &upper_triangular_matmul_shared, "Upper triangular matrix multiplication with shared memory");
}
