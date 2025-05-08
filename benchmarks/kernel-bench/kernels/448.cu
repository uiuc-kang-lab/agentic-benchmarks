#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 512
#define UNROLL_FACTOR 8

template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    const int64_t M,
    const int64_t K)
{
    const int64_t row = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t lane = tid % WARP_SIZE;

    scalar_t sum = 0;
    const scalar_t* row_ptr = A + row * K;

    #pragma unroll UNROLL_FACTOR
    for (int64_t k = tid; k < K; k += BLOCK_SIZE) {
        sum += row_ptr[k] * B[k];
    }

    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        C[row] = sum;
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous().view({-1});

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    
    auto C = torch::zeros({M, 1}, A.options());

    dim3 threads(BLOCK_SIZE);
    dim3 blocks(M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel<scalar_t><<<blocks, threads>>>(
            A_contig.data_ptr<scalar_t>(),
            B_contig.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K
        );
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}