#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matvec_mul_warp_reduce_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int64_t M,
    int64_t K)
{
    const int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (row >= M) return;

    scalar_t sum = 0;
    for (int64_t k = threadIdx.x; k < K; k += blockDim.x) {
        sum += A[row * K + k] * B[k];
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    if (threadIdx.x == 0) {
        C[row] = sum;
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be CUDA tensor");
    A = A.contiguous();
    B = B.contiguous();

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    auto B_flat = B.view({-1});

    TORCH_CHECK(B_flat.size(0) == K, "B size mismatch");
    auto C = torch::zeros({M}, A.options());

    const dim3 block(32, 8);  // 32 threads/warp, 8 warps/block
    const dim3 grid((M + block.y - 1) / block.y);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", [&] {
        matvec_mul_warp_reduce_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B_flat.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K);
    });

    cudaDeviceSynchronize();
    return C.view({M, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}