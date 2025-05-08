#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template<typename T>
__inline__ __device__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <typename scalar_t>
__global__ void layernorm_warp_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

    const int instance_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    const int warps_per_block = blockDim.x / warpSize;

    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;

    using accscalar_t = at::acc_type<scalar_t, true>;
    
    extern __shared__ char smem[];
    accscalar_t* warp_sums = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* warp_sums_sq = warp_sums + warps_per_block;

    const int CHUNK_SIZE = 4;
    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;

    const int items_per_thread = (normalized_size + blockDim.x - 1) / blockDim.x;
    const int base_idx = tid * CHUNK_SIZE;

    #pragma unroll
    for (int i = 0; i < items_per_thread; i++) {
        const int chunk_start = base_idx + i * blockDim.x * CHUNK_SIZE;
        
        #pragma unroll
        for (int j = 0; j < CHUNK_SIZE; j++) {
            const int idx = chunk_start + j;
            if (idx < normalized_size) {
                accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
                local_sum += val;
                local_sum_sq += val * val;
            }
        }
    }

    accscalar_t warp_sum = warp_reduce_sum(local_sum);
    accscalar_t warp_sum_sq = warp_reduce_sum(local_sum_sq);

    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
        warp_sums_sq[warp_id] = warp_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        accscalar_t block_sum = (lane_id < warps_per_block) ? warp_sums[lane_id] : 0;
        accscalar_t block_sum_sq = (lane_id < warps_per_block) ? warp_sums_sq[lane_id] : 0;

        block_sum = warp_reduce_sum(block_sum);
        block_sum_sq = warp_reduce_sum(block_sum_sq);

        if (lane_id == 0) {
            warp_sums[0] = block_sum;
            warp_sums_sq[0] = block_sum_sq;
        }
    }
    __syncthreads();

    const accscalar_t mean = warp_sums[0] / normalized_size;
    const accscalar_t variance = warp_sums_sq[0] / normalized_size - mean * mean;
    const accscalar_t inv_std = rsqrtf(variance + static_cast<accscalar_t>(eps));

    const int thread_stride = blockDim.x;
    #pragma unroll 4
    for (int idx = tid; idx < normalized_size; idx += thread_stride) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        accscalar_t normalized = (val - mean) * inv_std;
        out_ptr[idx] = static_cast<scalar_t>(
            normalized * static_cast<accscalar_t>(weight[idx]) + 
            static_cast<accscalar_t>(bias[idx]));
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    int normalized_size = weight.numel();
    int outer_size = x.numel() / normalized_size;

    const int warp_size = 32;
    const int num_warps = 8;
    const int threads = warp_size * num_warps;
    
    const int shared_mem_size = 2 * num_warps * sizeof(at::acc_type<float, true>);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_warp", ([&] {
        layernorm_warp_kernel<scalar_t><<<outer_size, threads, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            static_cast<float>(eps),
            output.data_ptr<scalar_t>(),
            normalized_size);
    }));

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward with warp optimizations (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}