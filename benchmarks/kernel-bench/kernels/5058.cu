#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Vector operations
template <typename scalar_t>
struct VectorOps {
    __device__ __forceinline__ static scalar_t square(scalar_t x) {
        return x * x;
    }
    
    __device__ __forceinline__ static scalar_t normalize(scalar_t x, scalar_t inv_norm) {
        return x * inv_norm;
    }
};

// Reduction operations
template <typename scalar_t>
struct ReductionOps {
    __device__ __forceinline__ static scalar_t warpReduce(scalar_t val) {
        #pragma unroll
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        return val;
    }

    __device__ __forceinline__ static scalar_t blockReduce(scalar_t val) {
        __shared__ scalar_t shared[32];
        int lane = threadIdx.x & 31;
        int wid = threadIdx.x >> 5;

        val = warpReduce(val);
        if (lane == 0) shared[wid] = val;
        __syncthreads();

        val = (threadIdx.x < (blockDim.x >> 5)) ? shared[lane] : 0;
        if (wid == 0) val = warpReduce(val);
        
        return val;
    }
};

// Main computation kernel
template <typename scalar_t>
__global__ void l2norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int C,
    const int stride_C,
    const int outer_stride
) {
    using VOps = VectorOps<scalar_t>;
    using ROps = ReductionOps<scalar_t>;

    const int vec_idx = blockIdx.x;
    const int base_offset = vec_idx * outer_stride;
    
    // Phase 1: Compute squared sum
    scalar_t thread_sum = 0;
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const scalar_t val = input[base_offset + i * stride_C];
        thread_sum += VOps::square(val);
    }
    
    // Reduce within block
    thread_sum = ROps::blockReduce(thread_sum);
    
    __shared__ scalar_t norm_val;
    if (threadIdx.x == 0) {
        norm_val = rsqrt(thread_sum + scalar_t(1e-12));
    }
    __syncthreads();
    
    // Phase 2: Normalize
    #pragma unroll 4
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        const int idx = base_offset + i * stride_C;
        output[idx] = VOps::normalize(input[idx], norm_val);
    }
}

torch::Tensor forward(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 2, "Input must have at least 2 dimensions");

    const int C = input.size(1);
    const int total_vectors = input.numel() / C;
    const int stride_C = input.stride(1);
    const int outer_stride = input.stride(0);

    auto output = torch::empty_like(input);

    const int threads = 512;
    const int blocks = total_vectors;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2norm_kernel", ([&] {
        l2norm_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Modular optimized L2 normalization");
}