#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 32
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

template <typename scalar_t>
__global__ void matmul_cuda_kernel(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B,
                                   scalar_t* __restrict__ C, int M, int K, int N) {
    const int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    const int col = blockIdx.x * TILE_WIDTH + threadIdx.x;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.y;
    
    scalar_t value = 0;

    // Process K dimension in tiles
    for (int t = 0; t < K; t += TILE_WIDTH) {
        scalar_t a_elem = 0;
        scalar_t b_elem = 0;
        
        if (row < M && (t + lane_id) < K) {
            a_elem = A[row * K + t + lane_id];
        }
        
        if (col < N && (t + warp_id) < K) {
            b_elem = B[(t + warp_id) * N + col];
        }

        // Perform dot product using warp shuffle
        #pragma unroll
        for (int i = 0; i < WARP_SIZE; ++i) {
            scalar_t a_broadcast = __shfl_sync(FULL_MASK, a_elem, i);
            scalar_t b_broadcast = __shfl_sync(FULL_MASK, b_elem, i);
            value += a_broadcast * b_broadcast;
        }
    }

    // Write result
    if (row < M && col < N) {
        C[row * N + col] = value;
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

    dim3 threads_per_block(WARP_SIZE, WARP_SIZE);
    dim3 num_blocks((N + TILE_WIDTH - 1) / TILE_WIDTH, (M + TILE_WIDTH - 1) / TILE_WIDTH);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matmul_cuda_kernel", ([&] {
        matmul_cuda_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, K, N);
    }));

    cudaDeviceSynchronize();
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &module_fn, "Matrix multiplication forward with warp shuffle (CUDA)");
}