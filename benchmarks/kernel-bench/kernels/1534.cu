#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARPS_PER_BLOCK 8
#define THREADS_PER_WARP 32

__global__ void matmul_shuffle_kernel(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C, int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0.0f;
    
    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    for (int t = 0; t < num_tiles; ++t) {
        // Load a tile of A into shared memory
        int a_col = t * TILE_SIZE + threadIdx.x;
        if(row < N && a_col < N)
            s_A[threadIdx.y][threadIdx.x] = A[row * N + a_col];
        else
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
            
        // Each thread loads its corresponding element of B
        int b_row = t * TILE_SIZE + threadIdx.y;
        float b_val = 0.0f;
        if(b_row < N && col < N)
            b_val = B[b_row * N + col];
        
        __syncthreads();
        
        // Compute dot product using warp shuffle to broadcast B values
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a_val = s_A[threadIdx.y][k];
            float b_shuffled = __shfl_sync(0xffffffff, b_val, k, 32);
            value = fmaf(a_val, b_shuffled, value);
        }
        
        __syncthreads();
    }
    
    if(row < N && col < N)
        C[row * N + col] = value;
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    const int N = A.size(0);
    
    auto C = torch::zeros({N, N}, A.options());
    
    dim3 block(THREADS_PER_WARP * WARPS_PER_BLOCK, TILE_SIZE / WARPS_PER_BLOCK);
    dim3 grid((N + TILE_SIZE - 1)/TILE_SIZE, (N + TILE_SIZE - 1)/TILE_SIZE);
    
    matmul_shuffle_kernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication with Warp Shuffle Optimization");
}