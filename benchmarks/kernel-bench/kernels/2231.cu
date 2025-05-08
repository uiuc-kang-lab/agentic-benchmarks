#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_DIM 16

__global__ void tiledMatmulKernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int K, int M, int N) {
    int row = blockIdx.x * TILE_DIM + threadIdx.y;
    int col = blockIdx.y * TILE_DIM + threadIdx.x;

    float cValue = 0.0f;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; t++) {
        int k_base = t * TILE_DIM;
        
        // Load tiles into shared memory
        if (row < M && (k_base + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = A[(k_base + threadIdx.x) * M + row];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if ((k_base + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(k_base + threadIdx.y) * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Single sync point after loading shared memory
        __syncthreads();

        // Compute dot product for this tile
        if (row < M && col < N) {
            #pragma unroll
            for (int k = 0; k < TILE_DIM; k++) {
                cValue += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
        }
        
        // Single sync point before next iteration
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = cValue;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch: A and B must have the same first dimension (K)");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 grid((M + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    tiledMatmulKernel<<<grid, blockDim>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Compute C = A.T * B using tiled shared memory with reduced synchronization (CUDA)");
}