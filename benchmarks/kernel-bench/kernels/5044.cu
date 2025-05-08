#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

template <typename scalar_t>
__device__ inline scalar_t warpReduceSum(scalar_t val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__device__ inline scalar_t blockReduceSum(scalar_t val) {
    __shared__ scalar_t shared[32];
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    
    val = warpReduceSum(val);
    
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    int num_warps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < num_warps) ? shared[lane] : scalar_t(0);
    if (warp_id == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

template <typename scalar_t>
__global__ void l2_norm_reduce_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {

    const int vec_idx = blockIdx.x;
    const int base_offset = vec_idx * outer_stride;

    scalar_t partial_sum = 0;
    int start = threadIdx.x + blockIdx.y * blockDim.x;
    int stride = blockDim.x * gridDim.y;

    for (int i = start; i < C; i += stride) {
        scalar_t val = input[base_offset + i * stride_C];
        partial_sum += val * val;
    }

    scalar_t block_sum = blockReduceSum<scalar_t>(partial_sum);
    if (threadIdx.x == 0) {
        atomicAdd(&norms[vec_idx], block_sum);
    }
}

template <typename scalar_t>
__global__ void l2_norm_normalize_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const scalar_t* __restrict__ norms,
    const int C,
    const int stride_C,
    const int outer_stride) {

    const int vec_idx = blockIdx.x;
    const int base_offset = vec_idx * outer_stride;
    __shared__ scalar_t inv_norm_shared;

    if (threadIdx.x == 0) {
        scalar_t norm_val = sqrt(norms[vec_idx]) + static_cast<scalar_t>(1e-12);
        inv_norm_shared = static_cast<scalar_t>(1.0) / norm_val;
    }
    __syncthreads();

    scalar_t inv_norm = inv_norm_shared;
    int start = threadIdx.x + blockIdx.y * blockDim.x;
    int stride = blockDim.x * gridDim.y;

    for (int i = start; i < C; i += stride) {
        int index = base_offset + i * stride_C;
        output[index] = input[index] * inv_norm;
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
    int blocksPerVector = (C + threads - 1) / threads;
    dim3 grid(total_vectors, blocksPerVector);

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_reduce", ([&] {
        l2_norm_reduce_kernel<scalar_t><<<grid, threads>>>(
            input.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            C,
            stride_C,
            outer_stride
        );
    }));

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "l2_norm_normalize", ([&] {
        l2_norm_normalize_kernel<scalar_t><<<grid, threads>>>(
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
    m.def("forward", &forward, "L2 normalization with shared inv_norm computation");
}