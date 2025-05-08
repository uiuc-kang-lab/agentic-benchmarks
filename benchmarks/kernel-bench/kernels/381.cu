#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Declare constant memory for vector B
__constant__ float const_vec_B[32768];  // 32K floats = 128KB, adjust if needed

template <typename scalar_t>
__global__ void constant_mem_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K,
    const bool use_const_mem) {
    
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    
    __shared__ scalar_t warp_results[32];
    
    if (row < M) {
        scalar_t thread_sum = 0;
        
        if (use_const_mem) {
            // Use constant memory for B when possible
            #pragma unroll 4
            for (int k = tid; k < K; k += blockDim.x) {
                scalar_t a_val = A[row][k];
                scalar_t b_val = const_vec_B[k];
                thread_sum += a_val * b_val;
            }
        } else {
            // Fallback to global memory with vectorized loads
            const scalar_t* A_ptr = A.data();
            const scalar_t* B_ptr = B.data();
            
            if constexpr (sizeof(scalar_t) == 4) {
                using vec4_t = float4;
                const int vec_K = K / 4;
                const vec4_t* A_vec = reinterpret_cast<const vec4_t*>(A_ptr + row * K);
                const vec4_t* B_vec = reinterpret_cast<const vec4_t*>(B_ptr);
                
                #pragma unroll 2
                for (int i = tid; i < vec_K; i += blockDim.x) {
                    vec4_t a_val = __ldg(&A_vec[i]);
                    vec4_t b_val = __ldg(&B_vec[i]);
                    thread_sum += a_val.x * b_val.x + a_val.y * b_val.y +
                                a_val.z * b_val.z + a_val.w * b_val.w;
                }
                
                // Handle remaining elements
                int remain_start = vec_K * 4;
                for (int k = remain_start + tid; k < K; k += blockDim.x) {
                    thread_sum += __ldg(&A_ptr[row * K + k]) * __ldg(&B_ptr[k]);
                }
            } else {
                for (int k = tid; k < K; k += blockDim.x) {
                    thread_sum += __ldg(&A_ptr[row * K + k]) * __ldg(&B_ptr[k]);
                }
            }
        }
        
        // Warp-level reduction using shuffle
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        }
        
        if (lane == 0) {
            warp_results[wid] = thread_sum;
        }
        __syncthreads();
        
        // Final reduction by first warp
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

torch::Tensor constant_mem_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    
    A = A.contiguous();
    B = B.contiguous();
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    auto B_flat = B.view({-1});
    
    auto C = torch::zeros({M, 1}, A.options());
    
    // Determine if we can use constant memory
    bool use_const_mem = (K <= 32768 && B.scalar_type() == torch::kFloat32);
    
    if (use_const_mem) {
        // Copy B to constant memory
        cudaMemcpyToSymbol(const_vec_B, B_flat.data_ptr<float>(), 
                          K * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    }
    
    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "constant_mem_matvec_cuda", ([&] {
        constant_mem_matvec_kernel<scalar_t><<<M, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K,
            use_const_mem);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &constant_mem_matvec_cuda, "Constant Memory Matrix-Vector Multiplication (CUDA)");
}