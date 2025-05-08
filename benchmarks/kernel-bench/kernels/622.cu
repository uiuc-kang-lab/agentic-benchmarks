#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define TILE_WIDTH 32  // Aligned with warp size for better efficiency

template <typename scalar_t>
__global__ void matmul_warp_aligned_kernel(const scalar_t* __restrict__ A,
                                            const scalar_t* __restrict__ B,
                                            scalar_t* __restrict__ C,
                                            const int M, const int K, const int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];

    const int warp_row = blockIdx.y * TILE_WIDTH + (threadIdx.y & ~(WARP_SIZE-1));
    const int warp_col = blockIdx.x * TILE_WIDTH + (threadIdx.x & ~(WARP_SIZE-1));
    const int lane_id = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Pre-compute boundary conditions for the entire warp
    const bool valid_row_warp = (warp_row < M);
    const bool valid_col_warp = (warp_col < N);
    
    scalar_t sum = 0;
    
    // Process tiles
    #pragma unroll 4
    for (int tile = 0; tile < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++tile) {
        const int tile_offset = tile * TILE_WIDTH;
        
        // Collaborative loading of tiles using vectorized loads where possible
        if (lane_id < TILE_WIDTH) {
            // Load entire rows/columns when the warp is fully within bounds
            if (valid_row_warp && (tile_offset + threadIdx.x) < K) {
                sA[threadIdx.y][threadIdx.x] = __ldg(&A[warp_row * K + tile_offset + threadIdx.x]);
            } else {
                sA[threadIdx.y][threadIdx.x] = 0;
            }
            
            if (valid_col_warp && (tile_offset + threadIdx.y) < K) {
                sB[threadIdx.y][threadIdx.x] = __ldg(&B[(tile_offset + threadIdx.y) * N + warp_col]);
            } else {
                sB[threadIdx.y][threadIdx.x] = 0;
            }
        }
        
        __syncthreads();
        
        // Compute partial products - unrolled for better instruction-level parallelism
        if (valid_row_warp && valid_col_warp) {
            #pragma unroll 8
            for (int k = 0; k < TILE_WIDTH; k += 4) {
                sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
                sum += sA[threadIdx.y][k+1] * sB[k+1][threadIdx.x];
                sum += sA[threadIdx.y][k+2] * sB[k+2][threadIdx.x];
                sum += sA[threadIdx.y][k+3] * sB[k+3][threadIdx.x];
            }
        }
        
        __syncthreads();
    }
    
    // Write results - check at thread level only when warp-level check passes
    const int global_row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int global_col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    if (valid_row_warp && valid_col_warp) {
        if (global_row < M && global_col < N) {
            C[global_row * N + global_col] = sum;
        }
    }
}

torch::Tensor module_fn(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "Input tensor A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "Input tensor B must be a CUDA tensor");
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    const int64_t N = B.size(1);
    
    TORCH_CHECK(K == B.size(0), "Inner dimensions must match");
    
    auto C = torch::empty({M, N}, A.options());
    
    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_warp_aligned_kernel", ([&] {
        matmul_warp_aligned_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    }));
    
    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Warp-aligned matrix multiplication forward (CUDA)");
}