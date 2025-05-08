#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32

// Kernel optimized for balanced workload distribution
__global__ void load_balanced_triangular_mm(const float* __restrict__ A,
                                             const float* __restrict__ B,
                                             float* __restrict__ C,
                                             int N) {
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (row >= N || col >= N) return;

    // Early exit for full upper-triangular blocks
    if (blockIdx.y * TILE_SIZE + TILE_SIZE <= blockIdx.x * TILE_SIZE) {
        C[row * N + col] = 0;
        return;
    }

    float sum = 0;
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    const int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int m = 0; m < numTiles; ++m) {
        int kA = m * TILE_SIZE + threadIdx.x;
        int kB = m * TILE_SIZE + threadIdx.y;

        // Load tiles with bounds checking
        sA[threadIdx.y][threadIdx.x] = (kA < N && row >= kA) ? A[row * N + kA] : 0;
        sB[threadIdx.y][threadIdx.x] = (kB < N && kB >= col) ? B[kB * N + col] : 0;
        
        __syncthreads();

        // Compute valid k range for this tile
        int kStart = max(col, m * TILE_SIZE);
        int kEnd = min(row + 1, (m + 1) * TILE_SIZE);
        
        #pragma unroll
        for (int k = kStart - m * TILE_SIZE; k < kEnd - m * TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        if (m < numTiles - 1) __syncthreads();
    }

    C[row * N + col] = (row >= col) ? sum : 0;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    auto C = torch::empty_like(A);

    dim3 blks((N + TILE_SIZE-1)/TILE_SIZE, (N + TILE_SIZE-1)/TILE_SIZE);
    dim3 thds(TILE_SIZE, TILE_SIZE);
    
    load_balanced_triangular_mm<<<blks, thds>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "Kernel failed: ", cudaGetErrorString(err));
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Load Balanced Triangular Matrix Multiplication");}
