#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
    __shared__ scalar_t shared[32];
    const int lane = threadIdx.x & (warpSize - 1);
    const int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

template <typename scalar_t>
__global__ void l2_norm_coalesced_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {
    
    const int tid = threadIdx.x;
    const int vec_idx = blockIdx.x;
    const int base_offset = vec_idx * outer_stride;
    
    // Step 1: Compute squared sum with coalesced memory access
    scalar_t thread_sum = 0;
    
    // Each thread processes elements with stride equal to block size
    // This ensures coalesced memory access within warps
    #pragma unroll 4
    for (int i = tid; i < C; i += blockDim.x) {
        const scalar_t val = input[base_offset + i * stride_C];
        thread_sum += val * val;
    }
    
    // Reduce within block
    const scalar_t block_sum = blockReduceSum(thread_sum);
    
    // Write norm to global memory
    if (tid == 0) {
        norms[vec_idx] = block_sum;
    }
    __syncthreads();
    
    // Step 2: Normalize with coalesced writes
    const scalar_t inv_norm = __frsqrt_rn(norms[vec_idx] + 1e-12);
    
    // Each thread normalizes elements with stride equal to block size
    #pragma unroll 4
    for (int i = tid; i < C; i += blockDim.x) {
        const int idx = base_offset + i * stride_C;
        output[idx] = input[idx] * inv_norm;
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
    auto norms = torch::empty({total_vectors}, input.options());
    
    // Use power of 2 threads for better performance
    const int threads = 256;
    const int blocks = total_vectors;
    
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_coalesced", ([&] {
        l2_norm_coalesced_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "L2 normalization with coalesced memory access");
}