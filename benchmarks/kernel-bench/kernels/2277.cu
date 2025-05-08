#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define TILE_SIZE 32
#define WARPS_PER_BLOCK 4
#define THREADS_PER_WARP 32

__device__ __inline__ float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__global__ void matMulWarpShuffleKernel(const float* __restrict__ A,
                                        const float* __restrict__ B,
                                        float* __restrict__ C,
                                        int K, int M, int N) {
    int warp_id = threadIdx.x / THREADS_PER_WARP;
    int lane_id = threadIdx.x % THREADS_PER_WARP;
    
    int row = blockIdx.x * TILE_SIZE + warp_id * (TILE_SIZE / WARPS_PER_BLOCK);
    int col = blockIdx.y * TILE_SIZE + lane_id;

    __shared__ float s_A[WARPS_PER_BLOCK][TILE_SIZE + 1];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
        int tiled_k = t * TILE_SIZE;
        
        // Load A tile with warp-specific loading
        int a_col = tiled_k + lane_id;
        if (a_col < K && row < M) {
            s_A[warp_id][lane_id] = A[a_col * M + row];
        } else {
            s_A[warp_id][lane_id] = 0.0f;
        }

        // Load B tile with coalesced access
        int b_row = tiled_k + threadIdx.x / (TILE_SIZE / WARPS_PER_BLOCK);
        int b_col = col;
        if (b_row < K && b_col < N) {
            s_B[threadIdx.x % TILE_SIZE][threadIdx.x / TILE_SIZE] = B[b_row * N + b_col];
        }
        __syncthreads();

        // Compute partial sums using warp shuffles
        for (int k = 0; k < TILE_SIZE; ++k) {
            float a_val = s_A[warp_id][k];
            float b_val = s_B[k][lane_id];
            sum += a_val * b_val;
        }
        __syncthreads();
    }

    // Warp-level reduction
    sum = warpReduceSum(sum);

    // Write result
    if (lane_id == 0 && row < M && col < N) {
        C[row * N + col] = sum;
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32, "Inputs must be float32");

    int K = A.size(0);
    int M = A.size(1);
    int N = B.size(1);
    TORCH_CHECK(B.size(0) == K, "Dimension mismatch");

    auto C = torch::zeros({M, N}, A.options());

    dim3 block(THREADS_PER_WARP * WARPS_PER_BLOCK);
    dim3 grid((M + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    matMulWarpShuffleKernel<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), K, M, N);
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Matrix multiply with warp shuffle optimization");
}