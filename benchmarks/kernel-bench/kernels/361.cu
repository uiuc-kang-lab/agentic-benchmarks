#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void balanced_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int num_blocks = gridDim.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    
    __shared__ scalar_t warp_sums[8];  // Support up to 256 threads (8 warps)
    
    // Each block handles multiple rows for better load balancing
    for (int row = bid; row < M; row += num_blocks) {
        scalar_t sum = 0;
        
        // Grid-stride loop over K dimension
        #pragma unroll 4
        for (int col = tid; col < K; col += blockDim.x) {
            sum += A[row][col] * B[col];
        }
        
        // Warp-level reduction using shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        
        // First thread in each warp writes to shared memory
        if (lane_id == 0) {
            warp_sums[warp_id] = sum;
        }
        __syncthreads();
        
        // Final reduction by first warp
        if (warp_id == 0 && lane_id < (blockDim.x >> 5)) {
            sum = warp_sums[lane_id];
            
            // Final warp reduction
            #pragma unroll
            for (int offset = (blockDim.x >> 6); offset > 0; offset >>= 1) {
                sum += __shfl_down_sync(0xffffffff, sum, offset);
            }
            
            if (lane_id == 0) {
                C[row][0] = sum;
            }
        }
    }
}

torch::Tensor balanced_matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    
    auto A_cont = A.contiguous();
    auto B_cont = B.contiguous();
    
    const int64_t M = A.size(0);
    const int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    auto B_flat = B.view({-1});
    
    auto C = torch::zeros({M, 1}, A.options());
    
    // Dynamic thread/block configuration based on problem size
    const int threads_per_block = 256;
    const int num_blocks = min(int((M + 3) / 4), 1024); // Each block handles multiple rows
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "balanced_matvec_mul_cuda", ([&] {
        balanced_matvec_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            A_cont.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &balanced_matvec_mul_cuda, "Balanced Matrix-Vector Multiplication (CUDA)");
}