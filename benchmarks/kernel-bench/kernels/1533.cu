#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#define TILE_SIZE 64
typedef cooperative_groups::thread_block_tile<32> warp_tile;

__global__ void matmul_warp_reduce(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int N) {
    __shared__ float s_A[TILE_SIZE][TILE_SIZE + 1]; // Avoid bank conflicts
    __shared__ float s_B[TILE_SIZE][TILE_SIZE + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float val = 0.0f;
    float reg_A[TILE_SIZE / 32];
    float reg_B[TILE_SIZE / 32];

    for (int i = 0; i < (N + TILE_SIZE - 1) / TILE_SIZE; ++i) {
        // Load tiles with padding awareness
        int a_col = i * TILE_SIZE + tx;
        s_A[ty][tx] = (row < N && a_col < N) ? A[row * N + a_col] : 0.0f;

        int b_row = i * TILE_SIZE + ty;
        s_B[ty][tx] = (b_row < N && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        // Warp tile processing
        for (int k_base = 0; k_base < TILE_SIZE; k_base += 32) {
            #pragma unroll
            for (int k_idx = 0; k_idx < 32; k_idx++) {
                int k = k_base + k_idx;
                reg_A[0] = s_A[ty][k];
                reg_B[0] = s_B[k][tx];
                val += __fmul_rn(reg_A[0], reg_B[0]);
            }

            // Warp reduction every 8 steps
            if (k_base % 8 == 0) {
                auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
                for (int offset = 16; offset > 0; offset >>= 1) {
                    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
                }
            }
        }

        __syncthreads();
    }

    // Final write
    if (row < N && col < N) {
        auto warp = cooperative_groups::tiled_partition<32>(cooperative_groups::this_thread_block());
        if (warp.thread_rank() == 0) {
            C[row * N + col] = val;
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    TORCH_CHECK(A.size(0) == B.size(0), "Matrix dimensions mismatch");

    const int N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    dim3 block(TILE_SIZE, TILE_SIZE / 4);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaFuncSetCacheConfig(matmul_warp_reduce, cudaFuncCachePreferShared);
    matmul_warp_reduce<<<grid, block>>>(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-reduced matrix multiplication");
}