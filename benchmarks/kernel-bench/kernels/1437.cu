#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define WARP_SIZE 32

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             const int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    
    // Calculate base indices
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;
    
    // Pre-compute boundary conditions
    const bool valid_thread = (row < N && col < N);
    
    float value = 0.0f;
    
    // Calculate number of full tiles
    const int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    #pragma unroll 1
    for (int m = 0; m < num_tiles; ++m) {
        const int tile_idx = m * TILE_SIZE;
        
        // Collaborative loading of tiles using vectorized access where possible
        if (threadIdx.x < TILE_SIZE) {
            // Load full vectors when possible
            #pragma unroll
            for (int i = 0; i < TILE_SIZE; i += WARP_SIZE) {
                const int load_idx = threadIdx.x + i;
                if (load_idx < TILE_SIZE) {
                    // Load A tile
                    const int a_row = blockIdx.y * TILE_SIZE + load_idx;
                    const int a_col = tile_idx + threadIdx.x;
                    s_A[load_idx][threadIdx.x] = (a_row < N && a_col < N) ? A[a_row * N + a_col] : 0.0f;
                    
                    // Load B tile
                    const int b_row = tile_idx + load_idx;
                    const int b_col = blockIdx.x * TILE_SIZE + threadIdx.x;
                    s_B[load_idx][threadIdx.x] = (b_row < N && b_col < N) ? B[b_row * N + b_col] : 0.0f;
                }
            }
        }
        
        __syncthreads();
        
        if (valid_thread) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                value += s_A[ty][k] * s_B[k][tx];
            }
        }
        
        __syncthreads();
    }
    
    // Coalesced write back to global memory
    if (valid_thread) {
        C[row * N + col] = value;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "A and B must be 2D");
    TORCH_CHECK(A.size(0) == A.size(1), "A must be square");
    TORCH_CHECK(B.size(0) == B.size(1), "B must be square");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of same size");

    const int N = A.size(0);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA, A.get_device());
    auto C = torch::zeros({N, N}, options);

    const int threads_per_block = TILE_SIZE * TILE_SIZE;
    dim3 threads(threads_per_block);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<blocks, threads>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix Multiplication (CUDA)");
}