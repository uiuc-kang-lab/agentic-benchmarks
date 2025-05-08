#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_SIZE 32
#define BLOCK_SIZE 256

__global__ void matmul_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             const int M, const int K, const int N) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x % TILE_SIZE;
    const int ty = threadIdx.x / TILE_SIZE;
    const int bx = blockIdx.x * TILE_SIZE;
    const int by = blockIdx.y * TILE_SIZE;

    float sum = 0.0f;
    
    // Precompute bounds check
    const bool valid_thread = (by + ty < M) && (bx + tx < N);
    
    const int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    
    #pragma unroll 4
    for (int t = 0; t < num_tiles; ++t) {
        const int k_idx = t * TILE_SIZE;
        
        // Load A tile using predication
        const bool valid_A = (by + ty < M) && (k_idx + tx < K);
        As[ty][tx] = valid_A ? A[(by + ty) * K + k_idx + tx] : 0.0f;
        
        // Load B tile using predication
        const bool valid_B = (k_idx + ty < K) && (bx + tx < N);
        Bs[ty][tx] = valid_B ? B[(k_idx + ty) * N + bx + tx] : 0.0f;
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (valid_thread) {
        C[(by + ty) * N + bx + tx] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
    TORCH_CHECK(B.is_contiguous(), "B must be contiguous");

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    auto C = torch::zeros({M, N}, A.options());

    dim3 threads(BLOCK_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE,
              (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_kernel<<<grid, threads>>>(
        A.data_ptr<float>(),
        B.data_ptr<float>(),
        C.data_ptr<float>(),
        M, K, N
    );

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp divergence optimized matrix multiplication (CUDA)");
}