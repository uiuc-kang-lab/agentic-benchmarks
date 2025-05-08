#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t blockReduceSum(scalar_t val) {
    static __shared__ scalar_t shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (wid == 0)
        val = warpReduceSum(val);
    return val;
}

template <typename scalar_t>
__global__ void l2_normalize_balanced_kernel(
    const scalar_t* input,
    scalar_t* output,
    scalar_t* norms,
    const int C,
    const int total_vectors,
    const int stride_C,
    const int outer_stride) {
    
    __shared__ scalar_t smem[32];
    const int vector_idx = blockIdx.x * blockDim.y + threadIdx.y;
    if (vector_idx >= total_vectors) return;

    scalar_t sum = 0;
    const int base_offset = vector_idx * outer_stride;
    const int tid = threadIdx.x;

    for (int i = tid; i < C; i += blockDim.x) {
        scalar_t val = input[base_offset + i * stride_C];
        sum += val * val;
    }

    scalar_t block_sum = blockReduceSum<scalar_t>(sum);
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        norms[vector_idx] = block_sum;
    }
    __syncthreads();

    scalar_t inv_norm = 1.0 / (sqrt(norms[vector_idx]) + 1e-12);
    for (int i = tid; i < C; i += blockDim.x) {
        output[base_offset + i * stride_C] = input[base_offset + i * stride_C] * inv_norm;
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

    const int threads_x = 256;
    const int threads_y = 1;
    const dim3 block(threads_x, threads_y);
    const dim3 grid((total_vectors + threads_y - 1) / threads_y);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_normalize_balanced", ([&] {
        l2_normalize_balanced_kernel<scalar_t><<<grid, block, 0>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            total_vectors,
            stride_C,
            outer_stride
        );
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Balanced L2 normalization");
}
