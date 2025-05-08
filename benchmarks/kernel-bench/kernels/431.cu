#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int TILE_SIZE>
__device__ __forceinline__ scalar_t warp_reduce(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename scalar_t, int TILE_SIZE>
__device__ __forceinline__ scalar_t compute_tile_sum(
    const scalar_t* __restrict__ A_row,
    const scalar_t* __restrict__ B,
    int64_t K,
    int tid) {
    scalar_t sum = 0;
    #pragma unroll 4
    for (int k = tid; k < K; k += blockDim.x * TILE_SIZE) {
        scalar_t partial = 0;
        #pragma unroll
        for (int t = 0; t < TILE_SIZE && (k + t * blockDim.x) < K; t++)
            partial += A_row[k + t * blockDim.x] * B[k + t * blockDim.x];
        sum += partial;
    }
    return sum;
}

template <typename scalar_t, int TILE_SIZE>
__global__ void matvec_mul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int64_t M,
    int64_t K) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (row >= M) return;

    const scalar_t* A_row = A + row * K;
    scalar_t sum = compute_tile_sum<scalar_t, TILE_SIZE>(A_row, B, K, tid);
    sum = warp_reduce<scalar_t, TILE_SIZE>(sum);

    if (threadIdx.x == 0)
        C[row] = sum;
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    
    A = A.contiguous();
    B = B.contiguous();

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "B size mismatch");
    auto C = torch::zeros({M}, A.options());

    constexpr int TILE_SIZE = 4;
    const int threads = 256;
    const dim3 blocks(M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", [&] {
        matvec_mul_kernel<scalar_t, TILE_SIZE><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K);
    });

    return C.view({M, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}