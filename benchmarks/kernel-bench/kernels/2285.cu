#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 32

__global__ void matMulCoalescedKernel(const float* __restrict__ A,
                                      const float* __restrict__ B,
                                      float* __restrict__ C,
                                      int K, int M, int N) {
    int row = blockIdx.x * TILE_SIZE + threadIdx.x;
    int col = blockIdx.y * TILE_SIZE + threadIdx.y;

    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        // Coalesced load of A tile (transposed)
        int loadA_col = t * TILE_SIZE + threadIdx.y;
        if (row < M && loadA_col < K)
            tileA[threadIdx.x][threadIdx.y] = A[loadA_col * M + row];
        else
            tileA[threadIdx.x][threadIdx.y] = 0.0f;

        // Coalesced load of B tile
        int loadB_row = t * TILE_SIZE + threadIdx.x;
        if (col < N && loadB_row < K)
            tileB[threadIdx.x][threadIdx.y] = B[loadB_row * N + col];
        else
            tileB[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        // Compute partial product with optimal shared memory access pattern
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[threadIdx.x][k] * tileB[k][threadIdx.y];
        }

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");

    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matMulCoalescedKernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced matrix multiplication with transposed A");
}
