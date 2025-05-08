#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>


#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// This kernel uses double buffering in shared memory to overlap global memory loads with computation.
__global__ void db_shared_matmul_kernel(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K) {
    // Global row and column indices
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    // Declare two shared memory buffers for A and B tiles
    __shared__ float As[2][TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[2][TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    // p will toggle between 0 and 1 for double buffering
    int p = 0;
    int t = 0;

    // Preload the first tile (tile 0) into buffer p
    int tiledCol = t * TILE_SIZE + threadIdx.x;
    int tiledRow = t * TILE_SIZE + threadIdx.y;
    if (row < M && tiledCol < K) {
        As[p][threadIdx.y][threadIdx.x] = A[row * K + tiledCol];
    } else {
        As[p][threadIdx.y][threadIdx.x] = 0.0f;
    }
    if (tiledRow < K && col < N) {
        Bs[p][threadIdx.y][threadIdx.x] = B[tiledRow * N + col];
    } else {
        Bs[p][threadIdx.y][threadIdx.x] = 0.0f;
    }

    __syncthreads();

    // Loop over remaining tiles with double buffering
    for (t = 0; t < numTiles - 1; t++) {
        int nextTile = t + 1;
        // Prefetch next tile into the alternate buffer (1 - p)
        int nextTiledCol = nextTile * TILE_SIZE + threadIdx.x;
        int nextTiledRow = nextTile * TILE_SIZE + threadIdx.y;
        if (row < M && nextTiledCol < K) {
            As[1 - p][threadIdx.y][threadIdx.x] = A[row * K + nextTiledCol];
        } else {
            As[1 - p][threadIdx.y][threadIdx.x] = 0.0f;
        }
        if (nextTiledRow < K && col < N) {
            Bs[1 - p][threadIdx.y][threadIdx.x] = B[nextTiledRow * N + col];
        } else {
            Bs[1 - p][threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();  // Ensure the prefetched tile is ready

        // Compute partial product using the current buffer p
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[p][threadIdx.y][k] * Bs[p][k][threadIdx.x];
        }

        __syncthreads(); // Ensure all threads finished computation before swapping buffers
        p = 1 - p; // Swap buffers
    }

    // Compute contribution from the last loaded tile in buffer p
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; k++) {
        sum += As[p][threadIdx.y][k] * Bs[p][k][threadIdx.x];
    }

    // Write result to global memory if within bounds
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// Host function to launch the kernel
void matrix_multiply_cuda(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_INPUT(C);

    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1);

    const float* d_A = A.data_ptr<float>();
    const float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    db_shared_matmul_kernel<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}

// PyTorch forward interface
torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    int M = A.size(0);
    int N = B.size(1);
    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device()).requires_grad(false);
    torch::Tensor C = torch::empty({M, N}, options);
    
    matrix_multiply_cuda(A, B, C);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Double Buffered Shared Memory Matrix Multiplication (CUDA)");
}
