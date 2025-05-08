#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix-vector multiplication with optimized memory access
template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int64_t M,
    int64_t K,
    int64_t A_stride)
{
    for (int64_t row = blockIdx.x * blockDim.x + threadIdx.x; row < M; row += blockDim.x * gridDim.x)
    {
        scalar_t sum = 0;
        const scalar_t* row_A = A + row * A_stride;
        
        // Use vectorized loads when possible for 128-bit alignment
        int64_t k = 0;
        #pragma unroll
        for (; k + 4 <= K; k += 4)
        {
            sum += __ldg(&row_A[k]) * __ldg(&B[k]);
            sum += __ldg(&row_A[k+1]) * __ldg(&B[k+1]);
            sum += __ldg(&row_A[k+2]) * __ldg(&B[k+2]);
            sum += __ldg(&row_A[k+3]) * __ldg(&B[k+3]);
        }
        
        // Handle remaining elements
        for (; k < K; ++k)
        {
            sum += __ldg(&row_A[k]) * __ldg(&B[k]);
        }
        
        C[row] = sum;
    }
}

// C++ function that wraps the CUDA kernel
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    int64_t A_stride = A.stride(0);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});

    auto C = torch::zeros({M, 1}, A.options());

    // Optimize block size for better occupancy
    int threads = 256;
    int blocks = (M + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel<scalar_t><<<blocks, threads>>>(
            A.data_ptr<scalar_t>(),
            B_flat.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M,
            K,
            A_stride);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}