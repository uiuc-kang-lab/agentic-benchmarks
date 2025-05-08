#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void hybrid_optimized_matvec_kernel(
    const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> B,
    torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> C,
    const int64_t M,
    const int64_t K) {

    // Grid-stride loop over rows for better workload distribution
    for (int row = blockIdx.x; row < M; row += gridDim.x) {
        int tid = threadIdx.x;
        int lane = tid & 31;
        int wid = tid >> 5;

        __shared__ scalar_t warp_results[32];
        scalar_t thread_sum = 0;

        // Get raw pointers for vectorized access
        const scalar_t* A_ptr = A.data();
        const scalar_t* B_ptr = B.data();

        // Vectorized loads for aligned data
        if constexpr (sizeof(scalar_t) == 4) {
            using vec_t = float4;
            int num_vec = K / 4;
            
            const vec_t* A_vec = reinterpret_cast<const vec_t*>(A_ptr + row * K);
            const vec_t* B_vec = reinterpret_cast<const vec_t*>(B_ptr);

            // Stride loop over vectors
            for (int i = tid; i < num_vec; i += blockDim.x) {
                vec_t a_val = __ldg(&A_vec[i]);
                vec_t b_val = __ldg(&B_vec[i]);
                thread_sum += a_val.x * b_val.x + a_val.y * b_val.y + 
                            a_val.z * b_val.z + a_val.w * b_val.w;
            }
            
            // Handle remaining elements
            int offset = num_vec * 4;
            for (int i = offset + tid; i < K; i += blockDim.x) {
                scalar_t a_val = __ldg(&A_ptr[row * K + i]);
                scalar_t b_val = __ldg(&B_ptr[i]);
                thread_sum += a_val * b_val;
            }
        } else {
            // Fallback for non-float types with __ldg
            for (int i = tid; i < K; i += blockDim.x) {
                scalar_t a_val = __ldg(&A_ptr[row * K + i]);
                scalar_t b_val = __ldg(&B_ptr[i]);
                thread_sum += a_val * b_val;
            }
        }

        // Warp-level reduction using shuffle
        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            thread_sum += __shfl_down_sync(mask, thread_sum, offset);
        }

        if (lane == 0) {
            warp_results[wid] = thread_sum;
        }
        __syncthreads();

        // Final reduction by first warp
        if (wid == 0 && lane < (blockDim.x >> 5)) {
            scalar_t final_sum = warp_results[lane];
            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                final_sum += __shfl_down_sync(mask, final_sum, offset);
            }
            if (lane == 0) {
                C[row][0] = final_sum;
            }
        }
    }
}

torch::Tensor hybrid_optimized_matvec_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    A = A.contiguous();
    B = B.contiguous();
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    auto B_flat = B.view({-1});
    auto C = torch::zeros({M, 1}, A.options());

    int threads = 256;
    int blocks = std::min(256, (int)M); // Adjust block count based on problem size

    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "hybrid_optimized_matvec_cuda", ([&] {
        hybrid_optimized_matvec_kernel<scalar_t><<<blocks, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M, K);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &hybrid_optimized_matvec_cuda, "Hybrid Optimized Matrix-Vector Multiplication (CUDA)");
}