#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void shared_memory_reduction_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {
    
    extern __shared__ scalar_t shared_mem[];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        scalar_t thread_sum = 0;

        // Load A and B using grid-stride loop
        for (int k = tid; k < K; k += blockDim.x) {
            thread_sum += A[row][k] * __ldg(&B[k]);
        }

        // Store partial results in shared memory
        shared_mem[tid] = thread_sum;
        __syncthreads();

        // Reduce within blocks using shared memory
        for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
            if (tid < stride) {
                shared_mem[tid] += shared_mem[tid + stride];
            }
            __syncthreads();
        }

        // Final reduction within warp using shuffle
        thread_sum = (tid < 32) ? shared_mem[tid] : 0;
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }

        // Write result for this block's row
        if (tid == 0) {
            C[row][0] = thread_sum;
        }
    }
}

torch::Tensor shared_memory_reduction_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    int threads = 256;
    int blocks = std::min((int)(M + threads - 1) / threads, 65535);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "shared_memory_reduction_matvec_cuda", ([&] {
        shared_memory_reduction_matvec_kernel<scalar_t><<<blocks, threads, threads * sizeof(scalar_t)>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_reduction_matvec_cuda, "Matrix-Vector Multiplication with Shared Memory Reduction (CUDA)");
}