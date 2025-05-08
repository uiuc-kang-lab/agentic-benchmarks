#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define BLOCK_SIZE 256

template <typename scalar_t>
__global__ void matvec_mul_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K)
{
    __shared__ scalar_t shared_mem[BLOCK_SIZE];
    
    const int64_t row = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t lane_id = tid % WARP_SIZE;
    
    if (row < M) {
        // Each thread computes partial sum for a segment
        scalar_t thread_sum = 0;
        for (int64_t k = tid; k < K; k += BLOCK_SIZE) {
            thread_sum += A[row][k] * B[k];
        }
        
        // First, perform warp-level reduction within each warp
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        // Each warp writes its reduced sum to shared memory
        if (lane_id == 0) {
            shared_mem[tid / WARP_SIZE] = thread_sum;
        }
        __syncthreads();
        
        // Let the first warp accumulate the warp sums
        if (tid < BLOCK_SIZE / WARP_SIZE) {
            thread_sum = shared_mem[tid];
            for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
                thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
            }
            if (tid == 0) {
                C[row][0] = thread_sum;
            }
        }
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B)
{
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1), "B must be a vector");

    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    dim3 threads(BLOCK_SIZE);
    dim3 blocks(M);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication (CUDA)");
}