#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel uses stride loops so that each thread can process multiple elements
// when the number of columns (K) exceeds the blockDim.x. Correct boundary handling is ensured
// by looping until all elements in the row are processed.

template <typename scalar_t>
__global__ void stride_loop_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {

    int row = blockIdx.x;
    if (row < M) {
        scalar_t sum = 0;
        
        // Stride loop: each thread handles a subset of the K dimension
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            sum += __ldg(&A[row][i]) * __ldg(&B[i]);
        }
        
        // Intra-warp reduction using shuffle operations
        for (int offset = 16; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // Use shared memory to accumulate results from different warps
        __shared__ scalar_t shared_sum[32]; // Assuming blockDim.x <= 1024 (max 32 warps)
        int lane = threadIdx.x & 31; // within-warp lane index
        int warpId = threadIdx.x >> 5; // warp index within the block
        if (lane == 0) {
            shared_sum[warpId] = sum;
        }
        __syncthreads();
        
        // Final reduction from each warp's result by the first warp in the block
        if (warpId == 0) {
            scalar_t final_sum = (threadIdx.x < (blockDim.x >> 5)) ? shared_sum[lane] : 0;
            for (int offset = 16; offset > 0; offset /= 2) {
                final_sum += __shfl_down_sync(0xffffffff, final_sum, offset);
            }
            if (lane == 0) {
                C[row][0] = final_sum;
            }
        }
    }
}

// C++ function wrapping the CUDA kernel

torch::Tensor stride_loop_matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());
    
    int threads = 256; // You can adjust this based on workload and GPU architecture
    // Launch one block per row of A
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "stride_loop_matvec_mul_cuda", ([&] {
        stride_loop_matvec_kernel<scalar_t><<<M, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K);
    }));
    
    return C;
}

// PyBind11 binding

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &stride_loop_matvec_mul_cuda, "Matrix-Vector Multiplication with Stride Loops (CUDA)");
}
