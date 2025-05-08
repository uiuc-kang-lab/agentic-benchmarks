#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define WARP_SIZE 32
#define BLOCK_WARPS 8
#define BLOCK_THREADS (WARP_SIZE * BLOCK_WARPS)
#define TILE_SIZE 32

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_FLOAT(x) TORCH_CHECK(x.scalar_type() == torch::kFloat32, #x " must be a float32 tensor")

__global__ void warp_matmul_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  const int N) {
    // Each thread processes 4 elements in the output row
    float reg_C[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    
    // Thread indexing
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int row = blockIdx.y * TILE_SIZE + warp_id * (TILE_SIZE/BLOCK_WARPS) + lane_id / 8;
    const int col_base = blockIdx.x * TILE_SIZE;

    // Shared memory for A tile
    __shared__ float As[TILE_SIZE][TILE_SIZE];

    // Process input matrix in tiles
    for (int tile = 0; tile < (N + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        // Collaborative loading of A tile
        if (row < N && tile * TILE_SIZE + lane_id < N) {
            As[warp_id * (TILE_SIZE/BLOCK_WARPS) + lane_id / 8][lane_id % TILE_SIZE] = 
                A[row * N + tile * TILE_SIZE + lane_id];
        }
        __syncthreads();

        // Process the tile
        if (row < N) {
            #pragma unroll
            for (int k = 0; k < TILE_SIZE && tile * TILE_SIZE + k < N; ++k) {
                float a_val = As[lane_id / 8][k];
                
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int col = col_base + lane_id * 4 + i;
                    if (col < N) {
                        float b_val = __ldg(&B[(tile * TILE_SIZE + k) * N + col]);
                        reg_C[i] += a_val * b_val;
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write results using warp shuffles for coalesced writes
    if (row < N) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int col = col_base + lane_id * 4 + i;
            if (col < N) {
                // Use warp shuffle to gather final results
                float final_val = reg_C[i];
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    final_val += __shfl_down_sync(0xffffffff, final_val, offset);
                }
                if (lane_id % 8 == 0) {
                    C[row * N + col] = final_val;
                }
            }
        }
    }
}

torch::Tensor forward(torch::Tensor A, torch::Tensor B) {
    CHECK_INPUT(A);
    CHECK_INPUT(B);
    CHECK_FLOAT(A);
    CHECK_FLOAT(B);

    TORCH_CHECK(A.dim() == 2 && A.size(0) == A.size(1), "A must be a square matrix");
    TORCH_CHECK(B.dim() == 2 && B.size(0) == B.size(1), "B must be a square matrix");
    TORCH_CHECK(A.size(0) == B.size(0), "A and B must be of the same size");

    int64_t N = A.size(0);
    auto C = torch::zeros({N, N}, A.options());

    const float* A_data = A.data_ptr<float>();
    const float* B_data = B.data_ptr<float>();
    float* C_data = C.data_ptr<float>();

    dim3 threads(BLOCK_THREADS);  // 256 threads per block
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    warp_matmul_kernel<<<blocks, threads>>>(A_data, B_data, C_data, N);
    C10_CUDA_CHECK(cudaGetLastError());

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Warp-optimized matrix multiplication kernel (CUDA)");
}