#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized CUDA kernel with improved thread and block indexing for better efficiency
// Each warp computes one output element (one row dot product) using coalesced memory access. 
// with threads in the warp loading consecutive elements from the row for coalesced access.

template <typename scalar_t>
__global__ void matvec_mul_kernel_optimized(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    const int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int row = blockIdx.x * blockDim.x / warp_size + threadIdx.x / warp_size;
    
    if (row < M) {
        scalar_t sum = 0;
        
        // Each thread in warp processes multiple elements with stride
        #pragma unroll 4
        for (int64_t col = lane; col < K; col += warp_size) {
            sum += A[row][col] * B[col];
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

// C++ function that wraps the CUDA kernel

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    A = A.contiguous();
    B = B.contiguous();

    int64_t M = A.size(0);
    int64_t K = A.size(1);

    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    // Use 256 threads per block (8 warps) for better occupancy
    const int threads_per_block = 256;
    const int blocks = (M * 32 + threads_per_block - 1) / threads_per_block;

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel_optimized<scalar_t><<<blocks, threads_per_block>>>(
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
    m.def("forward", &matvec_mul_cuda, "Optimized Matrix-Vector Multiplication with Improved Indexing (CUDA)");
}