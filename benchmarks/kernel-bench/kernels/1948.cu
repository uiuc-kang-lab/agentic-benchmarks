#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_SIZE 32

__global__ void optimized_triangular_mm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int N)
{
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    __shared__ float sA[TILE_SIZE][TILE_SIZE+1];
    __shared__ float sB[TILE_SIZE][TILE_SIZE+1];
    
    float sum = 0.0;

    if (row >= N || col > row) return;

    for (int tile = col; tile <= row; tile += TILE_SIZE) {
        int tile_end = min(tile + TILE_SIZE, row + 1);
        
        // Coalesced loading of A-tile
        if (threadIdx.x < tile_end - tile) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + (tile + threadIdx.x)];
        }
        
        // Coalesced loading of B-tile
        if (threadIdx.y < tile_end - tile) {
            sB[threadIdx.y][threadIdx.x] = B[(tile + threadIdx.y) * N + col];
        }
        __syncthreads();

        for (int k = 0; k < tile_end - tile; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    // Warp-level reduction
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x == 0) {
        C[row * N + col] = (col <= row) ? sum : 0.0f;
    }
}

at::Tensor forward(at::Tensor A, at::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 blocks((N + TILE_SIZE-1)/TILE_SIZE, (N + TILE_SIZE-1)/TILE_SIZE);
    dim3 threads(TILE_SIZE, TILE_SIZE);
    
    optimized_triangular_mm<<<blocks, threads>>>(A.data_ptr<float>(),
                                                B.data_ptr<float>(),
                                                C.data_ptr<float>(),
                                                N);
                                                
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized triangular MM (CUDA)");
}