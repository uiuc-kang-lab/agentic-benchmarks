#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that performs matrix-vector multiplication with optimized reductions.
// Each block processes one row of matrix A. Each thread computes a partial sum,
// then a warp-level reduction is performed with __shfl_down_sync(). 
// Warp leaders write their results into shared memory, and a final reduction across the warp sums
// produces the final output value for that row.

template <typename scalar_t>
__global__ void optimized_reduction_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {

    int row = blockIdx.x;  // one block per row
    if (row >= M) return;
    
    int tid = threadIdx.x;
    
    // Each thread computes a partial sum over elements in the row with a strided loop
    scalar_t sum = 0;
    for (int i = tid; i < K; i += blockDim.x) {
        sum += A[row][i] * B[i];
    }

    // Intra-warp reduction using warp shuffle primitives
    unsigned int mask = 0xFFFFFFFF; // all threads in the warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Use shared memory to collect the leading value from each warp
    __shared__ scalar_t shared[32]; // support up to 32 warps per block
    int warp_id = tid / warpSize;
    if ((tid & (warpSize - 1)) == 0) { // lane 0 of each warp
        shared[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction: first warp reduces the values stored in shared memory
    if (warp_id == 0) {
        int lane = tid & (warpSize - 1);
        // Only threads corresponding to the number of warps participate
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        scalar_t final_sum = (lane < numWarps) ? shared[lane] : static_cast<scalar_t>(0);
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            final_sum += __shfl_down_sync(mask, final_sum, offset);
        }
        if (lane == 0) {
            C[row][0] = final_sum;
        }
    }
}

// C++ interface function that launches the CUDA kernel
torch::Tensor optimized_reduction_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    
    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});

    // Allocate output tensor
    auto C = torch::zeros({M, 1}, A.options());

    // Define number of threads per block (e.g., 256) and one block per matrix row
    int threads = 256;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "optimized_reduction_matvec_cuda", ([&] {
        optimized_reduction_matvec_kernel<scalar_t><<<M, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K);
    }));

    return C;
}

// PyBind11 module binding
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_reduction_matvec_cuda, "Optimized Matrix-Vector Multiplication with Shared Memory and Warp-Level Reduction (CUDA)");
}
