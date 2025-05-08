#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel with warp-level cooperation for matrix-vector multiplication
// Each warp is responsible for computing one output element (one row dot product) 
// with threads in the warp loading consecutive elements from the row for coalesced access.

template <typename scalar_t>
__global__ void matvec_mul_kernel_coalesced(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    int64_t M,
    int64_t K) {

    __shared__ scalar_t shared_mem[32];  // Shared memory for partial sums
    
    // Each warp computes one row's dot product
    int warp_id = threadIdx.x / 32;           // warp index within the block
    int lane = threadIdx.x % 32;              // lane index within the warp
    int warpsPerBlock = blockDim.x / 32;         
    int row = blockIdx.x * warpsPerBlock + warp_id;  // global row index

    if (row < M) {
        scalar_t sum = 0;
        
        // Process multiple elements per thread for better instruction-level parallelism
        #pragma unroll 4
        for (int col = lane; col < K; col += 32) {
            scalar_t a = A[row][col];
            scalar_t b = B[col];
            sum = fma(a, b, sum);  // Use fused multiply-add
        }
        
        // Warp-level reduction using shuffle intrinsic
        // Unrolled for better performance
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 16);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 8);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 4);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 2);
        sum += __shfl_down_sync(0xFFFFFFFF, sum, 1);
        
        // First thread in each warp stores to shared memory
        if (lane == 0) {
            shared_mem[warp_id] = sum;
        }
        
        __syncthreads();
        
        // First warp reduces results from all warps
        if (warp_id == 0 && lane == 0) {
            C[row][0] = shared_mem[0];
        }
    }
}

// C++ function that wraps the CUDA kernel

torch::Tensor matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    // Ensure input tensors are on CUDA
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");

    // Ensure inputs are contiguous
    A = A.contiguous();
    B = B.contiguous();

    // Get dimensions
    int64_t M = A.size(0);
    int64_t K = A.size(1);

    // Check dimensions
    TORCH_CHECK(B.numel() == K, "B must have the same number of elements as columns in A");
    TORCH_CHECK(B.dim() == 1 || (B.dim() == 2 && B.size(1) == 1), "B must be a vector of shape (K,) or (K, 1)");

    // Flatten B to a 1D tensor
    auto B_flat = B.view({-1});

    // Allocate output tensor
    auto C = torch::zeros({M, 1}, A.options());

    // Set block and grid sizes
    // Use 128 threads per block (i.e., 4 warps per block)
    int threads = 128;
    int warpsPerBlock = threads / 32;
    int blocks = (M + warpsPerBlock - 1) / warpsPerBlock;

    // Dispatch based on data type
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "matvec_mul_cuda", ([&] {
        matvec_mul_kernel_coalesced<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));

    // Ensure synchronization
    cudaDeviceSynchronize();

    return C;
}

// PyBind11 binding code
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &matvec_mul_cuda, "Matrix-Vector Multiplication with Coalesced Memory Access (CUDA)");
}
