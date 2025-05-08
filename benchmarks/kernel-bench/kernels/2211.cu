#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define WARP_SIZE 32
#define TILE_M 16
#define TILE_N 16
#define BLOCK_K 32

__global__ void warpOptimizedMatMulKernel(const float* __restrict__ A,
                                         const float* __restrict__ B,
                                         float* __restrict__ C,
                                         int K, int M, int N) {
    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;
    const unsigned int warpM = laneId / 4;
    const unsigned int warpN = laneId % 4;
    
    int row = blockIdx.x * TILE_M + warpM;
    int col = blockIdx.y * TILE_N + warpN;

    // Register array for accumulating results
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Shared memory for cross-warp communication
    __shared__ float As[BLOCK_K][TILE_M];
    __shared__ float Bs[BLOCK_K][TILE_N];

    const uint32_t warp_mask = 0xFFFFFFFF;

    for (int k0 = 0; k0 < K; k0 += BLOCK_K) {
        // Collaborative loading using all threads in the block
        #pragma unroll 4
        for (int k = threadIdx.x; k < BLOCK_K * TILE_M; k += blockDim.x) {
            int t = k / TILE_M;
            int m = k % TILE_M;
            int global_k = k0 + t;
            int global_m = blockIdx.x * TILE_M + m;
            As[t][m] = (global_k < K && global_m < M) ? __ldg(&A[global_k * M + global_m]) : 0.0f;
        }

        #pragma unroll 4
        for (int k = threadIdx.x; k < BLOCK_K * TILE_N; k += blockDim.x) {
            int t = k / TILE_N;
            int n = k % TILE_N;
            int global_k = k0 + t;
            int global_n = blockIdx.y * TILE_N + n;
            Bs[t][n] = (global_k < K && global_n < N) ? __ldg(&B[global_k * N + global_n]) : 0.0f;
        }

        __syncthreads();

        // Compute using warp-level parallelism
        #pragma unroll
        for (int k = 0; k < BLOCK_K; k++) {
            float a_val = As[k][warpM];
            float b_val = Bs[k][warpN];
            
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                // Use warp shuffle to share values within the warp
                float a_shared = __shfl_sync(warp_mask, a_val, (laneId & ~3) + i, WARP_SIZE);
                float b_shared = __shfl_sync(warp_mask, b_val, (laneId & ~3) + i, WARP_SIZE);
                acc[i] += a_shared * b_shared;
            }
        }

        __syncthreads();
    }

    // Reduce within warp using shuffle operations
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        float sum = acc[i];
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(warp_mask, sum, offset);
        }
        
        // First thread in each warp writes result
        if (laneId == 0 && row + i < M && col < N) {
            C[(row + i) * N + col] = sum;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input B must be a CUDA tensor");
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Input A must be float32");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Input B must be float32");

    int K = A.size(0);
    int M = A.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");
    int N = B.size(1);

    auto C = torch::zeros({M, N}, torch::device(A.device()).dtype(A.dtype()));

    const int threadsPerBlock = 128; // 4 warps per block
    dim3 block(threadsPerBlock);
    dim3 grid((M + TILE_M - 1) / TILE_M, (N + TILE_N - 1) / TILE_N);

    const float* A_ptr = A.data_ptr<float>();
    const float* B_ptr = B.data_ptr<float>();
    float* C_ptr = C.data_ptr<float>();

    warpOptimizedMatMulKernel<<<grid, block>>>(A_ptr, B_ptr, C_ptr, K, M, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized matrix multiplication (CUDA)");
}