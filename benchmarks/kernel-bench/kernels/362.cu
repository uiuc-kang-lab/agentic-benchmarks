#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses grid-stride loops over rows and stride loops over columns
// to handle workloads larger than the available threads with correct boundary handling.

template <typename scalar_t>
__global__ void stride_loop_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    const int warpSize = 32;
    __shared__ scalar_t warp_sums[32];  // Enough for blockDim.x up to 1024

    // Use grid-stride loop for rows in case M is larger than gridDim.x
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        scalar_t sum = 0;
        
        // Each thread iterates over the K dimension in strides of blockDim.x
        for (int col = threadIdx.x; col < K; col += blockDim.x) {
            sum += A[row][col] * B[col];
        }
        
        // Warp-level reduction using shuffle intrinsics
        unsigned int mask = 0xffffffff;
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(mask, sum, offset);
        }
        
        // Each warp's lane 0 records the partial sum in shared memory
        int lane = threadIdx.x & (warpSize - 1);
        int warp_id = threadIdx.x / warpSize;
        if (lane == 0) {
            warp_sums[warp_id] = sum;
        }
        __syncthreads();
        
        // First thread of the block aggregates warp results
        if (threadIdx.x == 0) {
            scalar_t total = 0;
            int num_warps = (blockDim.x + warpSize - 1) / warpSize;
            for (int i = 0; i < num_warps; i++) {
                total += warp_sums[i];
            }
            C[row][0] = total;
        }
        __syncthreads(); // Ensure completion before processing next row
    }
}

// CUDA wrapper function

torch::Tensor stride_loop_matvec_cuda(torch::Tensor A, torch::Tensor B) {
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

    // Launch parameters: choose a fixed number of blocks for grid-stride looping over rows
    int threads = 256;
    int blocks = 256; // This can be tuned based on M

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "stride_loop_matvec_cuda", ([&] {
        stride_loop_matvec_kernel<scalar_t><<<blocks, threads>>>(
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
    m.def("forward", &stride_loop_matvec_cuda, "Matrix-Vector Multiplication with Stride Loops (CUDA)");
}
