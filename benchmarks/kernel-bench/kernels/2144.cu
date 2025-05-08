#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

__global__ void triangular_mm_kernel(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    if (row >= N || col >= N) return;
    
    float sum = 0.0f;
    
    int phase_limit = min(row + 1, N);
    for (int k_outer = 0; k_outer < phase_limit; k_outer += TILE_SIZE) {
        int k = k_outer + threadIdx.x;
        if (k <= row && k >= col) {
            As[threadIdx.y][threadIdx.x] = A[row * N + k];
            Bs[threadIdx.y][threadIdx.x] = B[k * N + col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k_inner = 0; k_inner < TILE_SIZE; ++k_inner) {
            sum += As[threadIdx.y][k_inner] * Bs[k_inner][threadIdx.x];
        }
        __syncthreads();
    }

    if (row >= col) {
        C[row * N + col] = sum;
    } else {
        C[row * N + col] = 0.0f;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    triangular_mm_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    cudaDeviceSynchronize();
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "Kernel execution failed");
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Triangular MM with shared memory");
}