#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void hybrid_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int wid = tid >> 5;
    
    // Use shared memory for larger reductions
    __shared__ scalar_t warp_results[32];
    
    if (row < M) {
        scalar_t thread_sum = 0;
        
        // Coalesced memory access with grid stride loop
        #pragma unroll 4
        for (int64_t k = tid; k < K; k += blockDim.x) {
            thread_sum += A[row][k] * B[k];
        }
        
        // First level reduction using warp shuffle
        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(mask, thread_sum, offset);
        }
        
        // Store warp results
        if (lane == 0) {
            warp_results[wid] = thread_sum;
        }
        __syncthreads();
        
        // Final reduction using first warp
        if (wid == 0) {
            scalar_t final_sum = (lane < (blockDim.x >> 5)) ? warp_results[lane] : 0;
            
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                final_sum += __shfl_down_sync(mask, final_sum, offset);
            }
            
            if (lane == 0) {
                C[row][0] = final_sum;
            }
        }
    }
}

torch::Tensor hybrid_matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    
    auto A_cont = A.contiguous();
    auto B_cont = B.contiguous();
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    auto B_flat = B.view({-1});
    
    auto C = torch::zeros({M, 1}, A.options());
    
    // Optimize block size based on matrix size
    int threads = (K > 1024) ? 256 : 128;
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "hybrid_matvec_mul_cuda", ([&] {
        hybrid_matvec_kernel<scalar_t><<<M, threads>>>(
            A_cont.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hybrid_matvec_mul_cuda, "Hybrid Matrix-Vector Multiplication (CUDA)");
}