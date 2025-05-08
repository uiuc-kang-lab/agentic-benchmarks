#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using warp-level shuffle reduction to minimize warp divergence
template <typename scalar_t>
__global__ void matvec_reduce_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t K) {

    // Each block processes one row
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    // Compute partial dot product using regular grid stride loop
    scalar_t sum = 0;
    for (int col = tid; col < K; col += blockSize) {
        // All threads perform the same iterations; no divergent branches
        sum += A[row][col] * B[col];
    }

    // Perform warp-level reduction using shuffle intrinsics
    // All threads in a warp execute the same code
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Now, thread lane 0 within each warp holds the reduced sum of that warp
    // Use shared memory to store these partial results
    __shared__ scalar_t warp_sums[32]; // maximum 32 warps per block (assuming blockDim.x <= 1024)
    int lane = tid & 31; // thread index modulo warp size (32)
    int warpId = tid >> 5; // equivalent to tid / 32

    if (lane == 0) {
        warp_sums[warpId] = sum;
    }
    __syncthreads();

    // Final reduction: let the first warp reduce the per-warp sums
    // Number of warps in this block
    int numWarps = blockSize >> 5; // blockDim.x / 32
    scalar_t final_sum = 0;
    if (tid < numWarps) {
        final_sum = warp_sums[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            final_sum += __shfl_down_sync(mask, final_sum, offset);
        }
        if (tid == 0) {
            C[row][0] = final_sum;
        }
    }
}

// C++ function that wraps the CUDA kernel
torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
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

    // Allocate output tensor
    auto C = torch::zeros({M, 1}, A.options());

    // Launch one block per row, using a block size that is a multiple of the warp size
    int threads = 128; // Should be a multiple of 32
    dim3 blocks(M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_reduce_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            K);
    }));

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA) with Warp Reduction");
}
