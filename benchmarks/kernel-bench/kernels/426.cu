#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix-vector multiplication
template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M)
    {
        // Compute partial sum in registers
        scalar_t local_sum = 0;
        
        // Coalesced memory access pattern for matrix A
        #pragma unroll 4
        for (int64_t k = 0; k < K; ++k)
        {
            local_sum += A[row][k] * B[k];
        }
        
        // Single write to global memory, no atomic needed
        C[row][0] = local_sum;
    }
}

// C++ function that wraps the CUDA kernel
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1), "B must be a vector");

    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    // Optimize thread block size for H100
    int threads = 512;
    int blocks = (M + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}