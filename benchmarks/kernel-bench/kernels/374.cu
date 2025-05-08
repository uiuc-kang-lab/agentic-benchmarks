#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// This kernel optimizes by using atomic operations only where necessary.
// Warp-level reductions are first performed using shuffle operations. 
// Then atomicAdd is used on a per-row basis to ensure correctness in reducing
// all warps' results into the output vector.
template <typename scalar_t>
__global__ void atomic_optimized_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {

    extern __shared__ scalar_t warp_results[];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    scalar_t thread_sum = 0;

    for (int i = tid; i < K; i += blockDim.x) {
        thread_sum += A[row][i] * B[i];
    }

    // Warp-level reduction using shuffle
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Store the result of each warp into shared memory
    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }
    __syncthreads();

    // Final reduction from shared memory to output using atomicAdd
    if (wid == 0) {
        thread_sum = (lane < (blockDim.x / warpSize)) ? warp_results[lane] : 0;
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(mask, thread_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(&C[row][0], thread_sum);
        }
    }
}

// C++ interface function wrapping the CUDA kernel
torch::Tensor atomic_optimized_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(B.numel() == K, "Dimension mismatch: B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    int threads = 256;
    int smem_size = (threads / warpSize) * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "atomic_optimized_matvec_cuda", ([&] {
        atomic_optimized_matvec_kernel<scalar_t><<<M, threads, smem_size>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K);
    }));
    
    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &atomic_optimized_matvec_cuda, "Atomic Optimized Matrix-Vector Multiplication (CUDA)");
}
