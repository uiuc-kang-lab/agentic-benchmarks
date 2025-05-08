#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for matrix-vector multiplication using warp-level primitives
// Each row is processed by one warp (32 threads). Each thread computes a partial product over a strided range of columns,
// and then we use __shfl_down_sync to reduce the partial sums within the warp.

template <typename scalar_t>
__global__ void matvec_warp_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) 
{
    // Each block processes one row
    int row = blockIdx.x;
    if (row < M) {
        // Assuming blockDim.x is 32 (i.e., one warp per row)
        int lane = threadIdx.x; // lane index within the warp
        scalar_t sum = 0;
        // Loop over the columns with stride equal to warp size
        for (int col = lane; col < K; col += warpSize) {
            sum += A[row][col] * B[col];
        }
        
        // Warp-level reduction using shuffle instructions
        // Use full warp mask (0xffffffff) since we assume active threads in warp
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // The first lane in the warp writes the result
        if (lane == 0) {
            C[row][0] = sum;
        }
    }
}

// C++ function that wraps the CUDA kernel

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure input tensors are on CUDA
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    // Get dimensions
    int64_t M = A.size(0);
    int64_t K = A.size(1);

    // Check dimensions for B
    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1), "B must be a vector of shape (K,) or (K, 1)");
    
    // Flatten B to be a 1D tensor if needed
    auto B_flat = B.view({-1});

    // Allocate output tensor
    auto C = torch::zeros({M, 1}, A.options());

    // Set block size to one warp (32 threads)
    int threads = 32;
    // One block per row
    int blocks = M;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_warp_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    // Ensure synchronization
    cudaDeviceSynchronize();

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication using warp-level reduction (CUDA)");
}
