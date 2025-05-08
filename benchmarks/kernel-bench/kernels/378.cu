#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void atomic_optimized_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {

    // Grid-stride loop for efficient workload distribution
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        int tid = threadIdx.x;
        scalar_t thread_sum = 0;

        // Use grid-stride loop to handle non-coalesced global memory accesses safely
        for (int i = tid; i < K; i += blockDim.x) {
            thread_sum += A[row][i] * B[i];
        }

        // Atomic add in shared memory
        __shared__ scalar_t shared_sum;
        if (tid == 0) shared_sum = 0;
        __syncthreads();

        atomicAdd(&shared_sum, thread_sum);
        __syncthreads();

        // Write result to global memory with atomicAdd
        if (tid == 0) {
            atomicAdd(&C[row][0], shared_sum);
        }
    }
}

torch::Tensor atomic_optimized_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());
    
    int threads = 256;
    // Set a grid size of 256, or max out with M, whichever is lower
    int blocks = std::min(256, (int)M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "atomic_optimized_matvec_cuda", ([&] {
        atomic_optimized_matvec_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &atomic_optimized_matvec_cuda, "Atomic Optimized Matrix-Vector Multiplication (CUDA)");
}
