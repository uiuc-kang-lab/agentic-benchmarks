#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matvec_mul_kernel_optimized(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {
    
    constexpr int warp_size = 32;
    const int warps_per_block = blockDim.x / warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    
    // Grid-stride loop for better load balancing
    for (int64_t row = blockIdx.x * warps_per_block + warp_id; row < M; row += gridDim.x * warps_per_block) {
        scalar_t sum = 0;
        
        // Coalesced access with unrolling
        #pragma unroll 4
        for (int64_t col = lane; col < K; col += warp_size) {
            sum += A[row][col] * B[col];
        }
        
        // Optimized warp reduction
        #pragma unroll
        for (int offset = warp_size/2; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        if (lane == 0) {
            C[row][0] = sum;
        }
    }
}

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must match A's columns");
    auto C = torch::zeros({M, 1}, A.options());

    // Optimized block/grid configuration
    const int threads_per_block = 256;  // 8 warps/block
    const int warps_per_block = threads_per_block / 32;
    
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int num_blocks = min(
        (M + warps_per_block - 1) / warps_per_block,
        num_sms * 4  // Increased blocks/SM for latency hiding
    );

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel_optimized<scalar_t><<<num_blocks, threads_per_block>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B.view({-1}).packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Optimized Matrix-Vector Multiplication (CUDA)");
}