#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void tiled_lower_tri_mm(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (row >= N || col >= N || col > row) {
        if (row < N && col < N && col > row) C[row*N+col] = 0.0f;
        return;
    }

    float sum = 0.0f;
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < num_tiles; ++t) {
        const int tiled_k = t * TILE_SIZE;
        const int load_row = tiled_k + threadIdx.x;
        const int load_col = tiled_k + threadIdx.y;

        if (row < N && load_row < N && (tiled_k + threadIdx.y) <= row)
            As[threadIdx.y][threadIdx.x] = A[row*N + load_row];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if (load_col < N && col < N && tiled_k >= col)
            Bs[threadIdx.y][threadIdx.x] = B[load_col*N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
            if ((tiled_k + k) > row || (tiled_k + k) < col) continue;
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col <= row)
        C[row*N+col] = sum;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);
    
    dim3 blocks((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    tiled_lower_tri_mm<<<blocks, threads>>>(A.data_ptr<float>(), 
                                          B.data_ptr<float>(),
                                          C.data_ptr<float>(),
                                          N);
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Tiled lower triangular matmul without atomics");
}