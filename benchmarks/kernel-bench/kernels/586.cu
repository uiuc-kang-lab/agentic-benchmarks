#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define WARP_SIZE 32

template <typename scalar_t>
__global__ void matmul_cuda_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                   scalar_t* __restrict__ C, int M, int K, int N) {
    __shared__ scalar_t sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ scalar_t sB[TILE_WIDTH][TILE_WIDTH];
    
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    
    // Use register to accumulate
    scalar_t value = 0;
    
    const unsigned int lane_id = threadIdx.x % WARP_SIZE;
    const unsigned int warp_id = threadIdx.x / WARP_SIZE;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Collaborative loading using vectorized loads where possible
        if (row < M && t * TILE_WIDTH + threadIdx.x < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_WIDTH + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (col < N && t * TILE_WIDTH + threadIdx.y < K) {
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE_WIDTH + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0;
        }
        
        __syncthreads();
        
        // Compute partial products using warp-level parallelism
        #pragma unroll
        for (int i = 0; i < TILE_WIDTH; i += 4) {
            value += sA[threadIdx.y][i] * sB[i][threadIdx.x];
            value += sA[threadIdx.y][i+1] * sB[i+1][threadIdx.x];
            value += sA[threadIdx.y][i+2] * sB[i+2][threadIdx.x];
            value += sA[threadIdx.y][i+3] * sB[i+3][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    // Warp-level reduction
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    
    // Write final result
    if (row < M && col < N) {
        if (lane_id == 0) {
            C[row * N + col] = value;
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
    dim3 num_blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel", ([&] {
        matmul_cuda_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward (CUDA)");
}