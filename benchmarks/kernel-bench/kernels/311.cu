#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 16
#define CHUNK_K 4

__global__ void bmm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int batch_size,
    const int M,
    const int K,
    const int N
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    const int row = by * TILE_SIZE + ty;
    const int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Base pointers for current batch
    const float* batch_A = A + bz * M * K;
    const float* batch_B = B + bz * K * N;
    
    // Main loop over tiles
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        // Load tiles into shared memory
        if (row < M && (tile + tx) < K) {
            As[ty][tx] = batch_A[row * K + tile + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && (tile + ty) < K) {
            Bs[ty][tx] = batch_B[(tile + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial dot products with manual unrolling
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; k += CHUNK_K) {
            sum += As[ty][k] * Bs[k][tx];
            sum += As[ty][k+1] * Bs[k+1][tx];
            sum += As[ty][k+2] * Bs[k+2][tx];
            sum += As[ty][k+3] * Bs[k+3][tx];
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row < M && col < N) {
        C[bz * M * N + row * N + col] = sum;
    }
}

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

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(
        (N + TILE_SIZE - 1) / TILE_SIZE,
        (M + TILE_SIZE - 1) / TILE_SIZE,
        batch_size
    );

    bmm_kernel<<<blocks, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        batch_size, M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_bmm, "Batched matrix multiplication (CUDA)");
}