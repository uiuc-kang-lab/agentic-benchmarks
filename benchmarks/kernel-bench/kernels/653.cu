#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32  // Increased tile width for better occupancy
#define MIN_BLOCKS_PER_SM 2  // Target multiple blocks per SM

template <typename scalar_t>
__global__ void matmul_hybrid_kernel(const scalar_t* __restrict__ A,
                                   const scalar_t* __restrict__ B,
                                   scalar_t* __restrict__ C,
                                   int M, int K, int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    // Grid-stride loop for better work distribution
    for (int tile_row = blockIdx.y * TILE_WIDTH; tile_row < M; tile_row += gridDim.y * TILE_WIDTH) {
        for (int tile_col = blockIdx.x * TILE_WIDTH; tile_col < N; tile_col += gridDim.x * TILE_WIDTH) {
            int row = tile_row + threadIdx.y;
            int col = tile_col + threadIdx.x;
            scalar_t sum = 0;

            // Process tiles in K dimension
            for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; t++) {
                int tiledACol = t * TILE_WIDTH + threadIdx.x;
                int tiledBRow = t * TILE_WIDTH + threadIdx.y;

                // Coalesced memory access pattern
                if (row < M && tiledACol < K)
                    sA[threadIdx.y][threadIdx.x] = A[row * K + tiledACol];
                else
                    sA[threadIdx.y][threadIdx.x] = 0;

                if (tiledBRow < K && col < N)
                    sB[threadIdx.y][threadIdx.x] = B[tiledBRow * N + col];
                else
                    sB[threadIdx.y][threadIdx.x] = 0;

                __syncthreads();

                // Unrolled inner loop for better instruction-level parallelism
                #pragma unroll
                for (int i = 0; i < TILE_WIDTH; i++) {
                    sum += sA[threadIdx.y][i] * sB[i][threadIdx.x];
                }

                // Only synchronize if not the last iteration
                if (t < (K + TILE_WIDTH - 1) / TILE_WIDTH - 1) {
                    __syncthreads();
                }
            }

            if (row < M && col < N) {
                C[row * N + col] = sum;
            }
        }
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t N = B.size(1);
    TORCH_CHECK(K == B.size(0), "Inner dimensions of A and B must match");

    auto C = torch::empty({M, N}, A.options());

    dim3 threads_per_block(TILE_WIDTH, TILE_WIDTH);
    
    // Calculate optimal grid dimensions based on device properties
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    
    int max_blocks_per_sm = props.maxThreadsPerMultiProcessor / (TILE_WIDTH * TILE_WIDTH);
    int target_blocks = props.multiProcessorCount * MIN_BLOCKS_PER_SM;
    
    dim3 num_blocks(
        min((N + TILE_WIDTH - 1) / TILE_WIDTH, (int)target_blocks),
        min((M + TILE_WIDTH - 1) / TILE_WIDTH, target_blocks)
    );

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_hybrid_kernel", ([&] {
        matmul_hybrid_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Optimized hybrid matrix multiplication (CUDA)");
}