#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32  // Increased tile width for better occupancy
#define BLOCK_ROW_WARPS 4
#define BLOCK_COL_WARPS 2
#define WARP_SIZE 32

template <typename scalar_t>
__device__ __forceinline__ void load_tile_aligned(const scalar_t* __restrict__ src,
                                                 scalar_t dst[TILE_WIDTH][TILE_WIDTH],
                                                 const int stride, const int M, const int N) {
    #pragma unroll
    for (int i = 0; i < TILE_WIDTH; i += WARP_SIZE) {
        if (threadIdx.y + i < M && threadIdx.x < N) {
            dst[threadIdx.y + i][threadIdx.x] = __ldg(&src[(threadIdx.y + i) * stride + threadIdx.x]);
        }
    }
}

template <typename scalar_t>
__global__ void matmul_cuda_kernel(const scalar_t* __restrict__ A,
                                 const scalar_t* __restrict__ B,
                                 scalar_t* __restrict__ C,
                                 const int M, const int K, const int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];
    
    // Register cache for partial results
    scalar_t thread_results[TILE_WIDTH/WARP_SIZE] = {0};
    
    const int warp_row = threadIdx.y / WARP_SIZE;
    const int warp_col = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    const int global_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int global_col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    for (int tile = 0; tile < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++tile) {
        // Collaborative loading using warps
        load_tile_aligned(A + blockIdx.y * TILE_WIDTH * K + tile * TILE_WIDTH,
                         sA, K, min(TILE_WIDTH, M - blockIdx.y * TILE_WIDTH),
                         min(TILE_WIDTH, K - tile * TILE_WIDTH));
                         
        load_tile_aligned(B + tile * TILE_WIDTH * N + blockIdx.x * TILE_WIDTH,
                         sB, N, min(TILE_WIDTH, K - tile * TILE_WIDTH),
                         min(TILE_WIDTH, N - blockIdx.x * TILE_WIDTH));
        
        __syncthreads();

        // Compute using register blocking
        #pragma unroll
        for (int k = 0; k < TILE_WIDTH; ++k) {
            #pragma unroll
            for (int i = 0; i < TILE_WIDTH/WARP_SIZE; ++i) {
                thread_results[i] += sA[threadIdx.y][k] * sB[k][threadIdx.x + i * WARP_SIZE];
            }
        }
        
        __syncthreads();
    }

    // Write results back to global memory
    #pragma unroll
    for (int i = 0; i < TILE_WIDTH/WARP_SIZE; ++i) {
        if (global_row < M && global_col + i * WARP_SIZE < N) {
            C[global_row * N + global_col + i * WARP_SIZE] = thread_results[i];
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

    dim3 threads(TILE_WIDTH, BLOCK_ROW_WARPS * WARP_SIZE);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel", ([&] {
        matmul_cuda_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Optimized modular matrix multiplication (CUDA)");
}