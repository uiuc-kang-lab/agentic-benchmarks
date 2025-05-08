#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that performs matrix-vector multiplication with aligned and coalesced accesses.
// Each block processes one row of A. The K dimension is processed in tiles of blockDim.x so that
// threads in each warp read consecutive (coalesced) memory locations from both A and B.

template <typename scalar_t>
__global__ void aligned_coalesced_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {

    // Each block handles one row of A.
    int row = blockIdx.x;
    scalar_t local_sum = 0;

    // Calculate number of tiles. Each tile has blockDim.x elements.
    int num_tiles = (K + blockDim.x - 1) / blockDim.x;

    // Loop over tiles. This ensures that within each tile, threads in a warp load consecutive
    // elements, resulting in coalesced accesses.
    for (int tile = 0; tile < num_tiles; tile++) {
        int col = tile * blockDim.x + threadIdx.x;
        if (col < K) {
            // Global memory accesses are coalesced: A[row][col] and B[col] are contiguous in memory.
            local_sum += A[row][col] * B[col];
        }
    }

    // Warp-level reduction using shuffle operations.
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Use shared memory to reduce partial sums across warps if blockDim.x > warpSize.
    __shared__ scalar_t warp_partial[32]; // enough for up to 1024 threads (32 warps)
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    
    if (lane == 0) {
        warp_partial[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction by the first warp.
    if (warp_id == 0) {
        // Only threads corresponding to the number of warps in the block participate.
        local_sum = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? warp_partial[lane] : 0;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
        }
        if (lane == 0) {
            C[row][0] = local_sum;
        }
    }
}

// C++ interface function wrapping the CUDA kernel
torch::Tensor aligned_coalesced_matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    
    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    // Choose block size (multiple of warpSize) to ensure coalesced accesses; one block per row.
    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "aligned_coalesced_matvec_mul_cuda", ([&] {
        aligned_coalesced_matvec_kernel<scalar_t><<<M, threads>>>(
            A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            M, K);
    }));

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &aligned_coalesced_matvec_mul_cuda, "Aligned Coalesced Matrix-Vector Multiplication (CUDA)");
}
