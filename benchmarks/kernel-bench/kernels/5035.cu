#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template<typename T>
__device__ __inline__ T warp_reduce(T val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template<typename T>
__device__ __inline__ T block_reduce(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warp_reduce(val);
    
    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    if (wid == 0) {
        val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
        val = warp_reduce(val);
    }
    return val;
}

template<typename scalar_t>
__global__ void l2_norm_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {
    
    const int vec_idx = blockIdx.x;
    const int base_offset = vec_idx * outer_stride;

    scalar_t sum = 0;
    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        scalar_t val = input[base_offset + i * stride_C];
        sum += val * val;
    }

    sum = block_reduce(sum);

    if (threadIdx.x == 0)
        atomicAdd(&norms[vec_idx], sum);

    __syncthreads();

    scalar_t inv_norm = 1.0 / (sqrt(norms[vec_idx]) + 1e-12);

    for (int i = threadIdx.x; i < C; i += blockDim.x) {
        output[base_offset + i * stride_C] = 
            input[base_offset + i * stride_C] * inv_norm;
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
    auto norms = torch::zeros({total_vectors}, input.options());

    const int threads = 512;
    const dim3 grid(total_vectors);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_optimized", [&] {
        l2_norm_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
    });

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized L2 normalization with single kernel");
}