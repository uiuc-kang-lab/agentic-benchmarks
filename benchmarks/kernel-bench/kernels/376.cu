#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
struct VectorizedLoad {
    using vec4_t = typename std::conditional<sizeof(scalar_t) == 4, float4, double2>::type;
    static constexpr int VEC_SIZE = sizeof(scalar_t) == 4 ? 4 : 2;
};

template <typename scalar_t>
__device__ __forceinline__ scalar_t vector_dot_product(
    const typename VectorizedLoad<scalar_t>::vec4_t& a,
    const typename VectorizedLoad<scalar_t>::vec4_t& b) {
    if constexpr (sizeof(scalar_t) == 4) {
        return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    } else {
        return a.x * b.x + a.y * b.y;
    }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t compute_partial_sum(
    const scalar_t* __restrict__ A_ptr,
    const scalar_t* __restrict__ B_ptr,
    const int64_t row,
    const int64_t K,
    const int tid,
    const int block_dim) {
    
    scalar_t thread_sum = 0;
    using vec4_t = typename VectorizedLoad<scalar_t>::vec4_t;
    constexpr int VEC_SIZE = VectorizedLoad<scalar_t>::VEC_SIZE;
    
    // Vectorized computation
    const vec4_t* A_vec = reinterpret_cast<const vec4_t*>(A_ptr + row * K);
    const vec4_t* B_vec = reinterpret_cast<const vec4_t*>(B_ptr);
    const int num_vec = K / VEC_SIZE;
    
    #pragma unroll 4
    for (int i = tid; i < num_vec; i += block_dim) {
        vec4_t a_val = __ldg(&A_vec[i]);
        vec4_t b_val = __ldg(&B_vec[i]);
        thread_sum += vector_dot_product<scalar_t>(a_val, b_val);
    }
    
    // Handle remaining elements
    const int offset = num_vec * VEC_SIZE;
    #pragma unroll
    for (int i = offset + tid; i < K; i += block_dim) {
        thread_sum += __ldg(&A_ptr[row * K + i]) * __ldg(&B_ptr[i]);
    }
    
    return thread_sum;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t warp_reduce_sum(scalar_t val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ void block_reduce_sum(
    scalar_t thread_sum,
    scalar_t* warp_results,
    const int lane,
    const int wid,
    scalar_t* final_result) {
    
    // Warp-level reduction
    thread_sum = warp_reduce_sum(thread_sum);
    
    // Store warp results
    if (lane == 0) {
        warp_results[wid] = thread_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (wid == 0) {
        thread_sum = (lane < (blockDim.x >> 5)) ? warp_results[lane] : 0;
        thread_sum = warp_reduce_sum(thread_sum);
        if (lane == 0) {
            *final_result = thread_sum;
        }
    }
}

template <typename scalar_t>
__global__ void modular_matvec_kernel(
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
        // Compute partial sum
        scalar_t thread_sum = compute_partial_sum(
            A.data(),
            B.data(),
            row,
            K,
            tid,
            blockDim.x
        );
        
        // Reduce and store result
        block_reduce_sum(
            thread_sum,
            warp_results,
            lane,
            wid,
            &C[row][0]
        );
    }
}

torch::Tensor modular_matvec_mul_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Input tensors must be CUDA tensors");
    
    A = A.contiguous();
    B = B.contiguous();
    
    int64_t M = A.size(0);
    int64_t K = A.size(1);
    
    TORCH_CHECK(B.numel() == K, "Dimension mismatch");
    auto B_flat = B.view({-1});
    
    auto C = torch::zeros({M, 1}, A.options());
    
    const int threads = 256;
    
    AT_DISPATCH_FLOATING_TYPES(A.scalar_type(), "modular_matvec_mul_cuda", ([&] {
        modular_matvec_kernel<scalar_t><<<M, threads>>>(
            A.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            B_flat.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            C.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
            M,
            K);
    }));
    
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &modular_matvec_mul_cuda, "Modular Matrix-Vector Multiplication (CUDA)");
}