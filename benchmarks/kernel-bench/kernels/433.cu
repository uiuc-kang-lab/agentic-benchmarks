#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matvec_warp_shuffle_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int64_t M,
    int64_t K)
{
    const int64_t row = blockIdx.x * blockDim.y + threadIdx.y;
    if (row >= M) return;

    const int64_t warp_start = threadIdx.x;  // Thread-specific start index
    scalar_t sum = 0;

    // Strided access for better coalescing
    for (int64_t k = warp_start; k < K; k += blockDim.x) {
        sum += A[row * K + k] * B[k];
    }

    // Full warp reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Write final result
    if (threadIdx.x == 0) {
        C[row] = sum;
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous().view({-1});

    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    auto C = torch::zeros({M}, A.options());

    // Configuration: 4 warps (rows) per block, 32 threads per warp
    const dim3 block(32, 4);  // threads x warps
    const dim3 grid((M + block.y - 1) / block.y);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", [&] {
        matvec_warp_shuffle_kernel<scalar_t><<<grid, block>>>(
            A.data_ptr<scalar_t>(),
            B.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K);
    });

    return C.view({M, 1});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Optimized Matrix-Vector Multiplication (CUDA)");
}