#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel using warp-level parallelism to minimize divergence
template <typename scalar_t>
__global__ void warp_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) 
{
    // Each warp computes one row of the output to ensure uniform control flow
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;

    // Only warps assigned a valid row perform computation
    if (warp_id < M) {
        scalar_t sum = 0;
        // Each thread in the warp processes a subset of the row's elements
        for (int k = lane; k < K; k += warpSize) {
            sum += A[warp_id][k] * B[k];
        }
        
        // Perform warp-level reduction without divergent branching
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Only one thread per warp writes the result
        if (lane == 0) {
            C[warp_id][0] = sum;
        }
    }
}

// C++ function that wraps the CUDA kernel
torch::Tensor warp_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1), "B must be a vector of shape (K,) or (K, 1)");

    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    // Each warp computes one row. With 256 threads per block, we have 256/32 = 8 warps per block.
    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (M + warps_per_block - 1) / warps_per_block;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "warp_matvec_cuda", ([&] {
        warp_matvec_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    // Ensure kernel completion before returning the result
    cudaDeviceSynchronize();
    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_matvec_cuda, "Warp optimized Matrix-Vector Multiplication (CUDA)");
}
