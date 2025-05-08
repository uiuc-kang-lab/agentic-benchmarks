#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void unrolled_hybrid_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    
    __shared__ scalar_t warp_results[32];
    
    if (row < M) {
        scalar_t thread_sum = 0;
        
        // Manual unrolling of the main computation loop
        const int64_t k_start = tid;
        const int64_t k_step = blockDim.x;
        const int64_t k_end = K - 4;  // Leave room for remainder handling
        
        // Main loop with manual unrolling by 4
        int64_t k = k_start;
        #pragma unroll 1
        for (; k < k_end; k += k_step * 4) {
            if (k + k_step * 3 < K) {
                scalar_t a0 = A[row][k];
                scalar_t a1 = A[row][k + k_step];
                scalar_t a2 = A[row][k + k_step * 2];
                scalar_t a3 = A[row][k + k_step * 3];
                
                scalar_t b0 = B[k];
                scalar_t b1 = B[k + k_step];
                scalar_t b2 = B[k + k_step * 2];
                scalar_t b3 = B[k + k_step * 3];
                
                thread_sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
            }
        }
        
        // Handle remaining elements
        #pragma unroll
        for (; k < K; k += k_step) {
            thread_sum += A[row][k] * B[k];
        }
        
        // Warp-level reduction using shuffle operations
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        // Store warp results
        if (lane == 0) {
            warp_results[wid] = thread_sum;
        }
        __syncthreads();
        
        // Final reduction using first warp
        if (wid == 0) {
            thread_sum = (lane < (blockDim.x >> 5)) ? warp_results[lane] : 0;
            
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
            }
            
            if (lane == 0) {
                C[row][0] = thread_sum;
            }
        }
    }
}

torch::Tensor unrolled_hybrid_matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    
    auto A_cont = A.contiguous();
    auto B_cont = B.contiguous();
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    auto B_flat = B.view({-1});
    
    auto C = torch::zeros({M, 1}, A.options());
    
    // Optimize thread count based on K dimension
    const int threads = (K >= 1024) ? 256 : ((K >= 512) ? 128 : 64);
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "unrolled_hybrid_matvec_mul_cuda", ([&] {
        unrolled_hybrid_matvec_kernel<scalar_t><<<M, threads>>>(
            A_cont.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &unrolled_hybrid_matvec_mul_cuda, "Unrolled Hybrid Matrix-Vector Multiplication (CUDA)");
}