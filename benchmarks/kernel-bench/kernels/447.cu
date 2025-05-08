#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 512
#define WARP_SIZE 32
#define WARPS_PER_BLOCK (BLOCK_SIZE / WARP_SIZE)

// CUDA kernel for matrix-vector multiplication using optimized shared memory reduction
template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int64_t M,
    const int64_t K) {

    // Shared memory to store per-warp partial sums
    __shared__ scalar_t warp_sums[WARPS_PER_BLOCK];

    int row = blockIdx.x; // Each block processes one row
    if (row >= M) return;

    int tid = threadIdx.x;
    scalar_t sum = 0;
    int row_offset = row * K;

    // Each thread computes a partial sum over elements in the row
    #pragma unroll
    for (int col = tid; col < K; col += BLOCK_SIZE) {
        sum += A[row_offset + col] * B[col];
    }

    // Intra-warp reduction using warp-level primitives
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write the result of each warp to shared memory
    if ((tid & (WARP_SIZE - 1)) == 0) {
        warp_sums[tid / WARP_SIZE] = sum;
    }
    __syncthreads();

    // Final reduction: first warp reduces the per-warp sums
    if (tid < WARPS_PER_BLOCK) {
        sum = warp_sums[tid];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (tid == 0) {
            C[row] = sum;
        }
    }
}

// C++ function wrapping the CUDA kernel
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous().view({-1});

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    // Allocate output tensor with M rows
    auto C = torch::zeros({M}, A.options());

    // Launch one block per row
    dim3 blocks(M);
    dim3 threads(BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K
        );
    }));

    // Return the result as an (M x 1) tensor
    return C.view({M, 1});
}

// PyBind11 module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}
