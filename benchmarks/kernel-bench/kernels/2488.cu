#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int BLOCK_SIZE = 32>
__global__ void matmul_transpose_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int M,
    const int N,
    const int K) {
    
    const unsigned FULL_MASK = 0xffffffff;
    const int warp_size = 32;
    const int warp_id = threadIdx.y / (warp_size/BLOCK_SIZE);
    const int lane_id = threadIdx.x % warp_size;
    
    const int row = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const int col = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    
    // Shared memory for cross-warp communication
    __shared__ scalar_t shared_sum[BLOCK_SIZE][BLOCK_SIZE/warp_size];
    
    scalar_t thread_sum = 0;
    
    // Process K dimension in chunks of warp_size
    for (int k_base = 0; k_base < K; k_base += warp_size) {
        scalar_t a_reg = 0, b_reg = 0;
        
        // Load values into registers
        if (k_base + lane_id < K) {
            if (row < M) {
                a_reg = A[(k_base + lane_id) * M + row];
            }
            if (col < N) {
                b_reg = B[col * K + k_base + lane_id];
            }
        }
        
        // Perform warp-level multiplication and reduction
        #pragma unroll
        for (int offset = 0; offset < warp_size; ++offset) {
            scalar_t a_shifted = __shfl_sync(FULL_MASK, a_reg, offset);
            scalar_t b_shifted = __shfl_sync(FULL_MASK, b_reg, offset);
            thread_sum = __fmaf_rn(a_shifted, b_shifted, thread_sum);
        }
    }
    
    // Store partial sums to shared memory
    if (lane_id < BLOCK_SIZE/warp_size) {
        shared_sum[threadIdx.x][lane_id] = thread_sum;
    }
    
    __syncthreads();
    
    // Reduce partial sums within the block
    if (lane_id == 0) {
        scalar_t block_sum = 0;
        #pragma unroll
        for (int i = 0; i < BLOCK_SIZE/warp_size; ++i) {
            block_sum += shared_sum[threadIdx.x][i];
        }
        
        // Write final result
        if (row < M && col < N) {
            C[row * N + col] = block_sum;
        }
    }
}

torch::Tensor matmul_transpose_cuda(torch::Tensor A, torch::Tensor B) {
    const int K = A.size(0);
    const int M = A.size(1);
    const int N = B.size(0);
    
    auto C = torch::empty({M, N}, A.options());
    
    constexpr int BLOCK_SIZE = 32;
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    AT_DISPATCH_FLOATING_TYPES(A.type(), "matmul_transpose_kernel", ([&] {
        matmul_transpose_kernel<scalar_t, BLOCK_SIZE><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K
        );
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matmul_transpose_cuda, "Warp-optimized matrix multiplication with transpose (CUDA)");
}