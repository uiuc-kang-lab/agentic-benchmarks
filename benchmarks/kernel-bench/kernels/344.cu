#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix-vector multiplication with loop unrolling
template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K)
{
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M)
    {
        scalar_t sum = 0;
        
        // Unroll the inner loop in chunks of 4
        int64_t k = 0;
        #pragma unroll 4
        for (; k + 3 < K; k += 4)
        {
            sum += A[row][k] * B[k];
            sum += A[row][k+1] * B[k+1];
            sum += A[row][k+2] * B[k+2];
            sum += A[row][k+3] * B[k+3];
        }
        
        // Handle remaining elements
        for (; k < K; ++k)
        {
            sum += A[row][k] * B[k];
        }
        
        C[row][0] = sum;
    }
}

// C++ function that wraps the CUDA kernel
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B)
{
    // Ensure input tensors are on CUDA
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    // Get dimensions
    int64_t M = A.size(0);
    int64_t K = A.size(1);

    // Check dimensions
    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1), "B must be a vector of shape (K,) or (K, 1)");

    // Flatten B to be a 1D tensor
    auto B_flat = B.view({-1});

    // Allocate output tensor
    auto C = torch::zeros({M, 1}, A.options());

    // Define block and grid sizes
    int threads = 256;
    int blocks = (M + threads - 1) / threads;

    // Dispatch based on data type
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

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}