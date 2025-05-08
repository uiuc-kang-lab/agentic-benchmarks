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
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int warp_id = tx / THREADS_PER_WARP;
    const int lane_id = tx % THREADS_PER_WARP;
    
    for (int row = blockIdx.y * TILE_SIZE + ty; row < N; row += gridDim.y * TILE_SIZE) {
        for (int col = blockIdx.x * TILE_SIZE + tx; col < N; col += gridDim.x * TILE_SIZE) {
            float value = 0.0f;
            
            for (int tile = 0; tile < (N + TILE_SIZE - 1)/TILE_SIZE; ++tile) {
                // Shared memory load for A
                int a_col = tile * TILE_SIZE + lane_id;
                if (row < N && a_col < N)
                    s_A[ty][lane_id] = A[row*N + a_col];
                else
                    s_A[ty][lane_id] = 0.0f;
                
                // Register storage for B
                int b_row = tile * TILE_SIZE + ty;
                float b_val = 0.0f;
                int b_col = col % TILE_SIZE;
                if (b_row < N && col < N)
                    b_val = B[b_row*N + col];
                
                __syncthreads();

                // Warp shuffle reduction for B values
                #pragma unroll
                for (int k = 0; k < TILE_SIZE; ++k) {
                    float a_val = s_A[ty][k];
                    float shuffled_b = __shfl_sync(0xffffffff, b_val, k, THREADS_PER_WARP);
                    value = fmaf(a_val, shuffled_b, value);
                }
                
                __syncthreads();
            }
            
            if (row < N && col < N)
                C[row*N + col] = value;
        }
    }
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