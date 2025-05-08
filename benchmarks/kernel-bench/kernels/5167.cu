#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t, int THREADS_PER_BLOCK=256, int VECTOR_SIZE=4>
__global__ void layernorm_hierarchical_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

    using accscalar_t = at::acc_type<scalar_t, true>;
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int lane_id = tid % warpSize;
    const int warp_id = tid / warpSize;
    constexpr int warps_per_block = THREADS_PER_BLOCK / warpSize;
    
    __shared__ accscalar_t warp_sums[warps_per_block];
    __shared__ accscalar_t warp_sums_sq[warps_per_block];
    
    const scalar_t* in_ptr = input + bid * normalized_size;
    scalar_t* out_ptr = output + bid * normalized_size;

    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;
    
    const int vector_stride = THREADS_PER_BLOCK * VECTOR_SIZE;
    const int vector_limit = normalized_size - (normalized_size % VECTOR_SIZE);
    
    #pragma unroll
    for (int idx = tid * VECTOR_SIZE; idx < vector_limit; idx += vector_stride) {
        accscalar_t vals[VECTOR_SIZE];
        
        #pragma unroll
        for (int v = 0; v < VECTOR_SIZE; v++) {
            vals[v] = static_cast<accscalar_t>(in_ptr[idx + v]);
            local_sum += vals[v];
            local_sum_sq += vals[v] * vals[v];
        }
    }
    
    for (int idx = vector_limit + tid; idx < normalized_size; idx += THREADS_PER_BLOCK) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
        warp_sums_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < warps_per_block) {
        local_sum = warp_sums[lane_id];
        local_sum_sq = warp_sums_sq[lane_id];
        
        #pragma unroll
        for (int offset = warps_per_block/2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
            local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
        }
    }

    __shared__ accscalar_t mean, inv_std;
    if (tid == 0) {
        mean = local_sum / normalized_size;
        accscalar_t var = (local_sum_sq / normalized_size) - (mean * mean);
        inv_std = rsqrt(var + static_cast<accscalar_t>(eps));
    }
    __syncthreads();

    #pragma unroll
    for (int idx = tid * VECTOR_SIZE; idx < vector_limit; idx += vector_stride) {
        accscalar_t vals[VECTOR_SIZE];
        accscalar_t weights[VECTOR_SIZE];
        accscalar_t biases[VECTOR_SIZE];
        
        #pragma unroll
        for (int v = 0; v < VECTOR_SIZE; v++) {
            vals[v] = static_cast<accscalar_t>(in_ptr[idx + v]);
            weights[v] = static_cast<accscalar_t>(weight[idx + v]);
            biases[v] = static_cast<accscalar_t>(bias[idx + v]);
            
            vals[v] = (vals[v] - mean) * inv_std;
            vals[v] = fma(vals[v], weights[v], biases[v]);
            
            out_ptr[idx + v] = static_cast<scalar_t>(vals[v]);
        }
    }

    for (int idx = vector_limit + tid; idx < normalized_size; idx += THREADS_PER_BLOCK) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        val = (val - mean) * inv_std;
        out_ptr[idx] = static_cast<scalar_t>(
            fma(val, static_cast<accscalar_t>(weight[idx]), 
                static_cast<accscalar_t>(bias[idx])));
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    const int normalized_size = weight.numel();
    const int outer_size = x.numel() / normalized_size;
    
    constexpr int threads = 256;
    const int blocks = outer_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        layernorm_hierarchical_kernel<scalar_t, threads><<<blocks, threads>>>(
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
    m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}