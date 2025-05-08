#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that performs matrix-vector multiplication using stride loops for better workload distribution.
template <typename scalar_t>
__global__ void stride_optimized_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {

    int row = blockIdx.x * blockDim.y + threadIdx.y;  // Each warp processes one row
    int tid = threadIdx.x;
    int lane = tid & 31;

    if (row < M) {
        scalar_t thread_sum = 0;
        
        // Use grid-stride loop for better workload distribution
        for (int k = tid; k < K; k += blockDim.x) {
            thread_sum += A[row][k] * B[k];
        }

        // Warp-level reduction using shuffle operations
        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(mask, thread_sum, offset);
        }

        // Use shared memory to store warp results
        __shared__ scalar_t warp_results[32];
        if (lane == 0) {
            warp_results[threadIdx.y] = thread_sum;
        }
        __syncthreads();

        // Final reduction by the first warp
        if (threadIdx.y == 0) {
            scalar_t final_sum = (lane < blockDim.y) ? warp_results[lane] : 0;
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                final_sum += __shfl_down_sync(mask, final_sum, offset);
            }
            if (lane == 0) {
                C[row][0] = final_sum;
            }
        }
    }
}

// C++ interface function wrapping the CUDA kernel
torch::Tensor stride_optimized_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(B.numel() == K, "Dimension mismatch: B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    int threads = 256;  // Use 256 threads per block
    int blocks = (M + threads - 1) / threads;  // Adjust block count based on problem size

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "stride_optimized_matvec_cuda", ([&] {
        stride_optimized_matvec_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K);
    }));
    
    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &stride_optimized_matvec_cuda, "Stride Optimized Matrix-Vector Multiplication (CUDA)");
}
