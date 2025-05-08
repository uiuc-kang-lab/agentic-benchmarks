#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void hybrid_adaptive_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K,
    const bool use_vectorized) {

    // Grid-stride loop over rows for better load balancing
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        const scalar_t* A_ptr = A.data() + row * K;
        const scalar_t* B_ptr = B.data();
        scalar_t thread_sum = 0;

        if (use_vectorized && sizeof(scalar_t) == 4) {  // float case with vectorization
            float4* A_vec = reinterpret_cast<float4*>(const_cast<scalar_t*>(A_ptr));
            float4* B_vec = reinterpret_cast<float4*>(const_cast<scalar_t*>(B_ptr));
            int num_vec = K / 4;

            // Vectorized processing with grid-stride
            for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
                float4 a_val = __ldg(&A_vec[i]);
                float4 b_val = __ldg(&B_vec[i]);
                thread_sum += a_val.x * b_val.x + a_val.y * b_val.y + 
                            a_val.z * b_val.z + a_val.w * b_val.w;
            }

            // Handle remaining elements
            int offset = num_vec * 4;
            for (int i = offset + threadIdx.x; i < K; i += blockDim.x) {
                thread_sum += __ldg(&A_ptr[i]) * __ldg(&B_ptr[i]);
            }
        } else {  // Non-vectorized case
            for (int i = threadIdx.x; i < K; i += blockDim.x) {
                thread_sum += __ldg(&A_ptr[i]) * __ldg(&B_ptr[i]);
            }
        }

        // Warp-level reduction using shuffle
        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(mask, thread_sum, offset);
        }

        // Store warp results in shared memory
        __shared__ scalar_t warp_results[32];
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;

        if (lane == 0) {
            warp_results[wid] = thread_sum;
        }
        __syncthreads();

        // Final reduction by first warp
        if (wid == 0 && lane == 0) {
            scalar_t final_sum = 0;
            int num_warps = (blockDim.x + 31) >> 5;
            for (int i = 0; i < num_warps; i++) {
                final_sum += warp_results[i];
            }
            C[row][0] = final_sum;
        }
    }
}

torch::Tensor hybrid_adaptive_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    // Adaptive launch configuration
    int threads = 256;
    int blocks = std::min(256, (int)((M + threads - 1) / threads));
    
    // Determine if vectorization should be used based on alignment and size
    bool use_vectorized = (K >= 128) && 
                         ((reinterpret_cast<uintptr_t>(A.data_ptr()) % 16) == 0) &&
                         ((reinterpret_cast<uintptr_t>(B.data_ptr()) % 16) == 0);

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "hybrid_adaptive_matvec_cuda", ([&] {
        hybrid_adaptive_matvec_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K, use_vectorized);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hybrid_adaptive_matvec_cuda, 
          "Hybrid Adaptive Matrix-Vector Multiplication (CUDA)");
}