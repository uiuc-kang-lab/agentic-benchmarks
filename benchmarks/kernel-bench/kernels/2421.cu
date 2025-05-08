#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    const int TILE_SIZE = 32;
    __shared__ scalar_t A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ scalar_t B_shared[TILE_SIZE][TILE_SIZE];
    
    // Block row and column
    const int blockRow = blockIdx.x;
    const int blockCol = blockIdx.y;
    
    // Thread row and column within tile
    const int threadRow = threadIdx.x;
    const int threadCol = threadIdx.y;
    
    // Global row and column indices
    const int row = blockRow * TILE_SIZE + threadRow;
    const int col = blockCol * TILE_SIZE + threadCol;
    
    scalar_t acc = 0;
    
    // Loop over tiles
    const int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int tile = 0; tile < numTiles; tile++) {
        // Load tile from A (transposed)
        if (row < M && (tile * TILE_SIZE + threadCol) < K) {
            A_shared[threadRow][threadCol] = A[(tile * TILE_SIZE + threadCol) * M + row];
        } else {
            A_shared[threadRow][threadCol] = 0;
        }
        
        // Load tile from B (transposed)
        if (col < N && (tile * TILE_SIZE + threadRow) < K) {
            B_shared[threadRow][threadCol] = B[col * K + (tile * TILE_SIZE + threadRow)];
        } else {
            B_shared[threadRow][threadCol] = 0;
        }
        
        __syncthreads();
        
        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += A_shared[threadRow][k] * B_shared[k][threadCol];
        }
        
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    const int TILE_SIZE = 32;
    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((M + TILE_SIZE - 1) / TILE_SIZE,
                (N + TILE_SIZE - 1) / TILE_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Matrix multiplication with transpose forward (CUDA)");
}