#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel using shared memory to cache b vector for improved performance

template <typename scalar_t>
__global__ void shared_memory_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {

    extern __shared__ scalar_t b_shared[];  // Dynamically allocated shared memory

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;

    scalar_t thread_sum = 0;

    // Load B into shared memory in blocks
    for (int i = tid; i < K; i += blockDim.x) {
        b_shared[i] = B[i];
    }
    __syncthreads();

    const scalar_t* A_ptr = A[row].data();
    
    for (int i = tid; i < K; i += blockDim.x) {
        scalar_t a_val = A_ptr[i];
        thread_sum += a_val * b_shared[i];
    }

    // Perform warp-level reduction using shuffle
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Aggregate results from different warps
    if (lane == 0) {
        atomicAdd(&C[row][0], thread_sum);
    }
}

torch::Tensor shared_memory_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    int threads = 256;
    size_t shared_mem_size = K * sizeof(scalar_t);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "shared_memory_matvec_cuda", ([&] {
        shared_memory_matvec_kernel<scalar_t><<<M, threads, shared_mem_size>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_memory_matvec_cuda, "Optimized Matrix-Vector Multiplication Using Shared Memory (CUDA)");
}
