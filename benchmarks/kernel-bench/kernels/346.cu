#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix-vector multiplication with coalesced memory access
template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K)
{
    __shared__ scalar_t shared_sum[256];
    
    int64_t row = blockIdx.x;
    int64_t tid = threadIdx.x;
    
    if (row < M) {
        // Each thread accumulates partial sum for strided elements
        scalar_t thread_sum = 0;
        
        // Stride through K dimension with multiple threads
        for (int64_t k = tid; k < K; k += blockDim.x) {
            thread_sum += A[row][k] * B[k];
        }
        
        // Store in shared memory
        shared_sum[tid] = thread_sum;
        __syncthreads();
        
        // Parallel reduction in shared memory
        for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                shared_sum[tid] += shared_sum[tid + stride];
            }
            __syncthreads();
        }
        
        // Write result
        if (tid == 0) {
            C[row][0] = shared_sum[0];
        }
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
    int threads = 256;  // Use full warps
    int blocks = M;     // One block per row

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