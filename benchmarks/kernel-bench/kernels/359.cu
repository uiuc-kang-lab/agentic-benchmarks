#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses shared memory for intra-warp results and warp-level shuffle operations for final reduction
// Each block processes one row of matrix A. Threads compute partial sums over the K dimension, then reduce within a warp,
// storing the result in shared memory. The first warp then reduces these warp sums to produce the final dot product.

template <typename scalar_t>
__global__ void warp_shared_kernel(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {

    int row = blockIdx.x; // One block per row
    int tid = threadIdx.x;
    scalar_t sum = 0;

    // Each thread accumulates partial sum from the input vector product over columns
    for (int k = tid; k < K; k += blockDim.x) {
        sum += A[row][k] * B[k];
    }

    // Intra-warp reduction using warp shuffle primitives
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Allocate shared memory for storing the result of each warp
    extern __shared__ scalar_t shared[];
    int warp_id = tid / warpSize;
    if ((tid & (warpSize - 1)) == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction: first warp loads warp-level sums from shared memory and reduces
    if (tid < (blockDim.x / warpSize)) {
        sum = shared[tid];
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (tid == 0) {
            C[row][0] = sum;
        }
    }
}

// C++ interface function
torch::Tensor warp_shared_red_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});

    // Allocate output tensor with shape (M, 1)
    auto C = torch::zeros({M, 1}, A.options());

    // Launch one block per row. Use a block size that is a multiple of warp size (e.g., 256 threads).
    int threads = 256;
    int blocks = M;

    // Shared memory: one element per warp
    size_t shared_mem_bytes = (threads / 32) * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "warp_shared_red_matvec_cuda", ([&] {
        shared_mem_bytes = (threads / 32) * sizeof(scalar_t);
        warp_shared_kernel<scalar_t><<<blocks, threads, shared_mem_bytes>>>(
            A.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
            M, K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &warp_shared_red_matvec_cuda, "Warp Shared Reduction Matrix-Vector Multiplication (CUDA)");
}
