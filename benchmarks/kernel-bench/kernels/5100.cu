#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t, typename accscalar_t>
__device__ void compute_statistics(
    const scalar_t* __restrict__ in_ptr,
    const int tid,
    const int normalized_size,
    const int blockDim_x,
    accscalar_t& local_sum,
    accscalar_t& local_sum_sq) {
    
    const int vector_size = 4;
    const int aligned_size = normalized_size / vector_size * vector_size;
    
    local_sum = 0;
    local_sum_sq = 0;
    
    for (int i = tid * vector_size; i < aligned_size; i += blockDim_x * vector_size) {
        float4 in_vec = *reinterpret_cast<const float4*>(&in_ptr[i]);
        accscalar_t vals[4] = {
            static_cast<accscalar_t>(in_vec.x),
            static_cast<accscalar_t>(in_vec.y),
            static_cast<accscalar_t>(in_vec.z),
            static_cast<accscalar_t>(in_vec.w)
        };
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            local_sum += vals[j];
            local_sum_sq += vals[j] * vals[j];
        }
    }

    for (int i = aligned_size + tid; i < normalized_size; i += blockDim_x) {
        accscalar_t val = static_cast<accscalar_t>(__ldg(&in_ptr[i]));
        local_sum += val;
        local_sum_sq += val * val;
    }
}

template <typename accscalar_t>
__device__ void reduce_statistics(
    accscalar_t* s_sum,
    accscalar_t* s_sum_sq,
    const int tid,
    const int blockDim_x) {
    
    for (int stride = blockDim_x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
}

template <typename scalar_t, typename accscalar_t>
__device__ void compute_normalized_output(
    const scalar_t* __restrict__ in_ptr,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out_ptr,
    const accscalar_t mean,
    const accscalar_t inv_std,
    const int tid,
    const int normalized_size,
    const int blockDim_x) {
    
    const int vector_size = 4;
    const int aligned_size = normalized_size / vector_size * vector_size;
    
    for (int i = tid * vector_size; i < aligned_size; i += blockDim_x * vector_size) {
        float4 in_vec = *reinterpret_cast<const float4*>(&in_ptr[i]);
        float4 w_vec = *reinterpret_cast<const float4*>(&weight[i]);
        float4 b_vec = *reinterpret_cast<const float4*>(&bias[i]);
        
        float4 out_vec;
        accscalar_t vals[4] = {in_vec.x, in_vec.y, in_vec.z, in_vec.w};
        accscalar_t w_vals[4] = {w_vec.x, w_vec.y, w_vec.z, w_vec.w};
        accscalar_t b_vals[4] = {b_vec.x, b_vec.y, b_vec.z, b_vec.w};
        
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            accscalar_t norm_val = (static_cast<accscalar_t>(vals[j]) - mean) * inv_std;
            reinterpret_cast<scalar_t*>(&out_vec)[j] = 
                static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w_vals[j]) + 
                                    static_cast<accscalar_t>(b_vals[j]));
        }
        
        *reinterpret_cast<float4*>(&out_ptr[i]) = out_vec;
    }

    for (int i = aligned_size + tid; i < normalized_size; i += blockDim_x) {
        scalar_t in_val = __ldg(&in_ptr[i]);
        scalar_t w_val = __ldg(&weight[i]);
        scalar_t b_val = __ldg(&bias[i]);
        accscalar_t norm_val = (static_cast<accscalar_t>(in_val) - mean) * inv_std;
        out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(w_val) + 
                                          static_cast<accscalar_t>(b_val));
    }
}

template <typename scalar_t>
__global__ void layernorm_forward_kernel_modular(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

    const int instance_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    const scalar_t* __restrict__ in_ptr = input + instance_idx * normalized_size;
    scalar_t* __restrict__ out_ptr = output + instance_idx * normalized_size;

    using accscalar_t = at::acc_type<scalar_t, true>;
    
    extern __shared__ char smem[];
    accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* s_sum_sq = s_sum + blockDim.x;

    accscalar_t local_sum, local_sum_sq;
    compute_statistics<scalar_t, accscalar_t>(in_ptr, tid, normalized_size, blockDim.x, 
                                            local_sum, local_sum_sq);

    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    reduce_statistics<accscalar_t>(s_sum, s_sum_sq, tid, blockDim.x);

    __shared__ accscalar_t mean;
    __shared__ accscalar_t inv_std;
    if (tid == 0) {
        mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
        accscalar_t var = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
        inv_std = static_cast<accscalar_t>(1) / sqrt(var + static_cast<accscalar_t>(eps));
    }
    __syncthreads();

    compute_normalized_output<scalar_t, accscalar_t>(in_ptr, weight, bias, out_ptr, 
                                                    mean, inv_std, tid, normalized_size, blockDim.x);
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    int normalized_size = weight.numel();
    int outer_size = x.numel() / normalized_size;

    int threads = std::min(((normalized_size + 31) / 32) * 32, 1024);
    int blocks = outer_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        int shared_size = threads * 2 * sizeof(accscalar_t);
        layernorm_forward_kernel_modular<scalar_t><<<blocks, threads, shared_size>>>(
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
    m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) modular",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}