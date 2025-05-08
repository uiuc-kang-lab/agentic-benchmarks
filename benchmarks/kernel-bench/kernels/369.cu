#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t, int BLOCK_SIZE=256, int WARP_SIZE=32>
__global__ void shared_warp_cascade_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {
    
    // Shared memory for partial sums - padded to avoid bank conflicts
    __shared__ scalar_t smem[BLOCK_SIZE + WARP_SIZE];
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    
    // Initialize shared memory
    smem[tid] = 0;
    
    if (row < M) {
        scalar_t thread_sum = 0;
        
        // Use vectorized loads for better memory throughput
        if constexpr (sizeof(scalar_t) == 4) {  // float
            const float4* A_vec = reinterpret_cast<const float4*>(A[row].data());
            const float4* B_vec = reinterpret_cast<const float4*>(B.data());
            const int vec_elements = 4;
            const int num_vectors = K / vec_elements;
            
            // Process 4 elements at a time
            for (int i = tid; i < num_vectors; i += BLOCK_SIZE) {
                float4 a = __ldg(&A_vec[i]);
                float4 b = __ldg(&B_vec[i]);
                thread_sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
            
            // Handle remaining elements
            const int remaining_start = num_vectors * vec_elements;
            for (int k = remaining_start + tid; k < K; k += BLOCK_SIZE) {
                thread_sum += A[row][k] * B[k];
            }
        } else {
            // Non-vectorized path for other types
            for (int k = tid; k < K; k += BLOCK_SIZE) {
                thread_sum += A[row][k] * B[k];
            }
        }
        
        // Store partial sum in shared memory
        smem[tid] = thread_sum;
    }
    __syncthreads();
    
    // First reduction step: block-level reduction in shared memory
    #pragma unroll
    for (int s = BLOCK_SIZE/2; s >= WARP_SIZE; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }
    
    // Second reduction step: warp-level reduction using shuffle
    if (tid < WARP_SIZE) {
        scalar_t warp_sum = smem[tid];
        
        // Reduce within warp using shuffle
        #pragma unroll
        for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
        }
        
        // First thread in warp writes result
        if (lane == 0) {
            smem[wid] = warp_sum;
        }
    }
    __syncthreads();
    
    // Final reduction: first warp reduces results from all warps
    if (tid < (BLOCK_SIZE / WARP_SIZE)) {
        scalar_t final_sum = smem[tid];
        
        // Final warp-level reduction
        #pragma unroll
        for (int offset = (BLOCK_SIZE/WARP_SIZE)/2; offset > 0; offset >>= 1) {
            final_sum += __shfl_down_sync(0xffffffff, final_sum, offset);
        }
        
        // Write final result
        if (tid == 0 && row < M) {
            C[row][0] = final_sum;
        }
    }
}

torch::Tensor shared_warp_cascade_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    
    auto A_cont = A.contiguous();
    auto B_cont = B.contiguous();
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    auto B_flat = B.view({-1});
    
    auto C = torch::zeros({M, 1}, A.options());
    
    constexpr int BLOCK_SIZE = 256;
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "shared_warp_cascade_matvec_cuda", ([&] {
        shared_warp_cascade_kernel<scalar_t, BLOCK_SIZE><<<M, BLOCK_SIZE>>>(
            A_cont.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_warp_cascade_matvec_cuda, "Matrix-Vector Multiplication with Shared Memory Cascade (CUDA)");
}