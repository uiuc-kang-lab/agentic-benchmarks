#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

template <typename scalar_t>
__global__ void matmul_reduced_sync_kernel(const scalar_t* __restrict__ A,
                                            const scalar_t* __restrict__ B,
                                            scalar_t* __restrict__ C,
                                            const int M, const int K, const int N) {
    __shared__ scalar_t As[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t Bs[TILE_WIDTH][TILE_WIDTH];
    
    const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    scalar_t sum = 0;

    // Calculate number of tiles needed
    const int numTiles = (K + TILE_WIDTH - 1) / TILE_WIDTH;
    
    for (int t = 0; t < numTiles; ++t) {
        const int tileIdx = t * TILE_WIDTH;
        
        // Load tiles into shared memory
        if (row < M && (tileIdx + threadIdx.x) < K) {
            As[threadIdx.y][threadIdx.x] = __ldg(&A[row * K + tileIdx + threadIdx.x]);
        } else {
            As[threadIdx.y][threadIdx.x] = 0;
        }
        
        if ((tileIdx + threadIdx.y) < K && col < N) {
            Bs[threadIdx.y][threadIdx.x] = __ldg(&B[(tileIdx + threadIdx.y) * N + col]);
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0;
        }
        
        // Single sync after both loads complete
        __syncthreads();
        
        // Compute dot product for this tile
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        
        // Single sync before next iteration to ensure shared memory is ready
        __syncthreads();
    }
    
    // Write result
    if (row < M && col < N) {
        C[row * N + col] = sum;
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
    
    const dim3 threads(TILE_WIDTH, TILE_WIDTH);
    const dim3 blocks((N + TILE_WIDTH - 1) / TILE_WIDTH,
                     (M + TILE_WIDTH - 1) / TILE_WIDTH);
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_reduced_sync_kernel", ([&] {
        matmul_reduced_sync_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Reduced sync matrix multiplication forward (CUDA)");
}