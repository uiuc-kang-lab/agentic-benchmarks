#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_unrolled_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

    int instance_idx = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;

    using accscalar_t = at::acc_type<scalar_t, true>;
    
    extern __shared__ char smem[];
    accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* s_sum_sq = s_sum + (blockDim.x >> 5);
    
    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;

    const int vector_size = 4;
    const int vector_limit = normalized_size - (normalized_size % vector_size);
    
    #pragma unroll 4
    for (int i = tid * vector_size; i < vector_limit; i += blockDim.x * vector_size) {
        accscalar_t vals[vector_size];
        #pragma unroll
        for (int j = 0; j < vector_size; j++) {
            vals[j] = static_cast<accscalar_t>(in_ptr[i + j]);
            local_sum += vals[j];
            local_sum_sq += vals[j] * vals[j];
        }
    }

    #pragma unroll
    for (int i = vector_limit + tid; i < normalized_size; i += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    if (lane == 0) {
        s_sum[warp_id] = local_sum;
        s_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();

    if (tid == 0) {
        accscalar_t final_sum = 0;
        accscalar_t final_sum_sq = 0;
        
        #pragma unroll
        for (int i = 0; i < (blockDim.x >> 5); ++i) {
            final_sum += s_sum[i];
            final_sum_sq += s_sum_sq[i];
        }

        accscalar_t mean = final_sum / normalized_size;
        accscalar_t variance = final_sum_sq / normalized_size - mean * mean;
        s_sum[0] = mean;
        s_sum[1] = rsqrt(variance + static_cast<accscalar_t>(eps));
    }
    __syncthreads();

    accscalar_t mean = s_sum[0];
    accscalar_t inv_std = s_sum[1];

    #pragma unroll 4
    for (int i = tid * vector_size; i < vector_limit; i += blockDim.x * vector_size) {
        accscalar_t vals[vector_size];
        accscalar_t weights[vector_size];
        accscalar_t biases[vector_size];

        #pragma unroll
        for (int j = 0; j < vector_size; j++) {
            vals[j] = static_cast<accscalar_t>(in_ptr[i + j]);
            weights[j] = static_cast<accscalar_t>(weight[i + j]);
            biases[j] = static_cast<accscalar_t>(bias[i + j]);
            
            vals[j] = (vals[j] - mean) * inv_std;
            vals[j] = vals[j] * weights[j] + biases[j];
            
            out_ptr[i + j] = static_cast<scalar_t>(vals[j]);
        }
    }

    #pragma unroll
    for (int i = vector_limit + tid; i < normalized_size; i += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
        val = (val - mean) * inv_std;
        out_ptr[i] = static_cast<scalar_t>(val * weight[i] + bias[i]);
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    int normalized_size = weight.numel();
    int outer_size = x.numel() / normalized_size;

    int threads = std::min(1024, ((normalized_size + 3) / 4 + 31) & ~31);
    int blocks = outer_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        int shared_mem_size = (threads >> 5) * 2 * sizeof(accscalar_t);
        layernorm_unrolled_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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