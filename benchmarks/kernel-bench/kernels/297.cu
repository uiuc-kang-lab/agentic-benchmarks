#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE 16

// Kernel for batched matrix multiplication using shared memory tiling with manual loop unrolling
// A: (batch_size, M, K), B: (batch_size, K, N), C: (batch_size, M, N)
__global__ void bmm_tiled_manual_unroll_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int batch_size,
    int M,
    int K,
    int N
) {
    // Determine the batch index from the z-dimension
    int b = blockIdx.z;
    int A_batch_offset = b * M * K;
    int B_batch_offset = b * K * N;
    int C_batch_offset = b * M * N;
    
    // Row and column index for C
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    
    // Shared memory tiles for A and B
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    
    float sum = 0.0f;
    // Calculate the number of tiles along the K dimension
    int numTiles = (K + TILE - 1) / TILE;

    #pragma unroll
    for (int t = 0; t < numTiles; t++) {
        int a_col = t * TILE + threadIdx.x;
        if (row < M && a_col < K) {
            As[threadIdx.y][threadIdx.x] = A[b * M * K + row * K + a_col];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        int b_row = t * TILE + threadIdx.y;
        if (b_row < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = B[b * K * N + b_row * N + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();

        // Manually unrolled inner loop for TILE=16
        sum += As[threadIdx.y][0]  * Bs[0][threadIdx.x];
        sum += As[threadIdx.y][1]  * Bs[1][threadIdx.x];
        sum += As[threadIdx.y][2]  * Bs[2][threadIdx.x];
        sum += As[threadIdx.y][3]  * Bs[3][threadIdx.x];
        sum += As[threadIdx.y][4]  * Bs[4][threadIdx.x];
        sum += As[threadIdx.y][5]  * Bs[5][threadIdx.x];
        sum += As[threadIdx.y][6]  * Bs[6][threadIdx.x];
        sum += As[threadIdx.y][7]  * Bs[7][threadIdx.x];
        sum += As[threadIdx.y][8]  * Bs[8][threadIdx.x];
        sum += As[threadIdx.y][9]  * Bs[9][threadIdx.x];
        sum += As[threadIdx.y][10] * Bs[10][threadIdx.x];
        sum += As[threadIdx.y][11] * Bs[11][threadIdx.x];
        sum += As[threadIdx.y][12] * Bs[12][threadIdx.x];
        sum += As[threadIdx.y][13] * Bs[13][threadIdx.x];
        sum += As[threadIdx.y][14] * Bs[14][threadIdx.x];
        sum += As[threadIdx.y][15] * Bs[15][threadIdx.x];

        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[b * M * N + row * N + col] = sum;
    }
}

// Forward function to launch the kernel
torch::Tensor forward_bmm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 3, "A must be 3D");
    TORCH_CHECK(B.dim() == 3, "B must be 3D");
    TORCH_CHECK(A.size(0) == B.size(0), "Batch sizes must match");
    TORCH_CHECK(A.size(2) == B.size(1), "Inner dimensions (K) must match");
    
    int batch_size = A.size(0);
    int M = A.size(1);
    int K = A.size(2);
    int N = B.size(2);

    auto options = torch::TensorOptions().dtype(A.dtype()).device(A.device());
    auto C = torch::zeros({batch_size, M, N}, options);

    dim3 threads(TILE, TILE);
    dim3 blocks((N + TILE - 1) / TILE, (M + TILE - 1) / TILE, batch_size);

    bmm_tiled_manual_unroll_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication with manual unrolling (CUDA)");
}
