#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void matvec_mul_kernel_aligned(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) 
{
    const int warp_size = 32;
    const int warp_id = threadIdx.x / warp_size;
    const int lane = threadIdx.x % warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    
    // Grid-stride loop over rows
    for (int64_t row = blockIdx.x * warps_per_block + warp_id; row < M; row += gridDim.x * warps_per_block) {
        scalar_t sum = 0;
        
        // Ensure 128-bit aligned access when possible
        const scalar_t* row_ptr = &A[row][0];
        
        // Process elements in chunks of 4 when possible for better memory coalescing
        int64_t col;
        #pragma unroll 4
        for (col = lane; col + 3 < K; col += warp_size) {
            // Use __ldg for read-only memory access
            scalar_t a0 = __ldg(row_ptr + col);
            scalar_t b0 = __ldg(&B[col]);
            sum += a0 * b0;
        }
        
        // Handle remaining elements
        if (col < K) {
            sum += __ldg(row_ptr + col) * __ldg(&B[col]);
        }
        
        // Warp-level reduction using shuffle
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

    // Ensure inputs are contiguous and properly aligned
    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});
    
    // Ensure output tensor is properly aligned
    auto options = A.options(); // No alignment method available
    auto C = torch::zeros({M, 1}, options);

    const int threads_per_block = 256;
    const int warps_per_block = threads_per_block / 32;
    
    int num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0);
    int blocks_per_sm = 2;
    int num_blocks = min(
        (static_cast<int>(M) + warps_per_block - 1) / warps_per_block,
        num_sms * blocks_per_sm
    );

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel_aligned<scalar_t><<<num_blocks, threads_per_block>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Aligned Matrix-Vector Multiplication (CUDA)");
}