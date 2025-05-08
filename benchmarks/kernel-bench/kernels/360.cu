#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that assigns one thread block per row, eliminating the need for global atomic operations.
// Each block computes the dot product for a single row with a grid-stride loop and warp-level reduction using shuffle intrinsics.

template <typename scalar_t>
__global__ void dedicated_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t K) {

    // Each block is responsible for one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    scalar_t sum = 0;
    // Grid-stride loop: each thread processes multiple elements along K
    for (int k = tid; k < K; k += blockSize) {
        sum += A[row][k] * B[k];
    }

    // Warp-level reduction using shuffle intrinsics
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Each warp's lane 0 holds the partial sum; store it in shared memory
    __shared__ scalar_t warpSums[32];
    int lane = tid & 31;
    int warpId = tid >> 5;  // equivalent to tid / 32
    if (lane == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();

    // Final reduction: let the first warp combine the results
    if (warpId == 0) {
        // Load partial sums; threads beyond the number of warps get zero
        sum = (tid < (blockSize + 31) / 32) ? warpSums[lane] : 0;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        if (lane == 0) {
            C[row][0] = sum;
        }
    }
}

// C++ interface function that launches one block per row, so that no atomic operations are needed

torch::Tensor dedicated_matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1), "B must be a vector of shape (K,) or (K, 1)");

    auto B_flat = B.view({-1});

    // Allocate output tensor
    auto C = torch::zeros({M, 1}, A.options());

    // Launch one block per row. This eliminates race conditions and avoids the need for atomics.
    int threads = 256;
    dim3 grid(M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "dedicated_matvec_mul_cuda", ([&] {
        dedicated_matvec_kernel<scalar_t><<<grid, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K);
    }));

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dedicated_matvec_mul_cuda, "Dedicated Block Matrix-Vector Multiplication (CUDA)");
}
