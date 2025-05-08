#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel for matrix-vector multiplication
// Combines grid-stride loop accumulation (for coalesced memory access) and warp-level shuffle reduction

template <typename scalar_t>
__global__ void optimized_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t K) {

    // Each block processes one row of the matrix
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    scalar_t sum = 0;

    // Use a grid-stride loop to cover the K dimension.
    // This guarantees coalesced accesses even for large K
    for (int col = tid; col < K; col += blockSize) {
        sum += A[row][col] * B[col];
    }

    // Perform warp-level reduction using shuffle intrinsics to reduce divergence
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Each warp now has its partial sum in lane 0. Store these in shared memory
    __shared__ scalar_t warpSums[32]; // Enough for up to 1024 threads (32 warps)
    int lane = tid & 31;        // thread index within its warp
    int warpId = tid >> 5;      // warp id within the block

    if (lane == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();

    // Final reduction: the first warp reduces all the per-warp sums
    if (tid < blockDim.x / 32) {
        sum = warpSums[lane]; // Each thread of first warp loads one partial sum
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (lane == 0) {
            C[row][0] = sum;
        }
    }
}

// C++ function wrapping the CUDA kernel
// This function ensures inputs are contiguous and on CUDA, performs type dispatch,
// and launches one block per row using an optimal block size

torch::Tensor optimized_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1),
                "B must be a vector of shape (K,) or (K, 1)");

    auto B_flat = B.view({-1});

    // Allocate the output tensor with shape (M, 1)
    auto C = torch::zeros({M, 1}, A.options());

    // Choose a block size as a multiple of warp size for efficient reduction
    int threads = 256; 
    dim3 blocks(M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "optimized_matvec_cuda", ([&] {
        optimized_matvec_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K);
    }));

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_matvec_cuda, "Optimized Matrix-Vector Multiplication (CUDA)");
}
