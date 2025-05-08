#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_uniform_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

    using accscalar_t = at::acc_type<scalar_t, true>;
    
    const unsigned int tid = threadIdx.x;
    const unsigned int bid = blockIdx.x;
    const unsigned int wid = tid >> 5;
    const unsigned int lane = tid & 31;
    
    const scalar_t* in_ptr = input + bid * normalized_size;
    scalar_t* out_ptr = output + bid * normalized_size;
    
    extern __shared__ char smem[];
    accscalar_t* warp_sums = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* warp_sums_sq = warp_sums + (blockDim.x >> 5);
    
    accscalar_t thread_sum = 0;
    accscalar_t thread_sum_sq = 0;

    const int vector_size = 4;
    const int vectors_per_thread = (normalized_size + (blockDim.x * vector_size - 1)) / (blockDim.x * vector_size);
    
    #pragma unroll 4
    for (int v = 0; v < vectors_per_thread; v++) {
        const int idx = tid * vector_size + v * blockDim.x * vector_size;
        if (idx < normalized_size - (vector_size - 1)) {
            accscalar_t vals[vector_size];
            
            #pragma unroll
            for (int i = 0; i < vector_size; i++) {
                vals[i] = static_cast<accscalar_t>(in_ptr[idx + i]);
                thread_sum += vals[i];
                thread_sum_sq += vals[i] * vals[i];
            }
        }
    }

    const int remainder_start = vectors_per_thread * blockDim.x * vector_size;
    if (remainder_start + tid < normalized_size) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[remainder_start + tid]);
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
        thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
    }
    thread_sum = __shfl_sync(0xffffffff, thread_sum, 0);
    thread_sum_sq = __shfl_sync(0xffffffff, thread_sum_sq, 0);

    if (lane == 0) {
        warp_sums[wid] = thread_sum;
        warp_sums_sq[wid] = thread_sum_sq;
    }
    __syncthreads();

    if (tid < 32) {
        accscalar_t final_sum = (tid < (blockDim.x >> 5)) ? warp_sums[tid] : 0;
        accscalar_t final_sum_sq = (tid < (blockDim.x >> 5)) ? warp_sums_sq[tid] : 0;

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum += __shfl_down_sync(0xffffffff, final_sum, offset);
            final_sum_sq += __shfl_down_sync(0xffffffff, final_sum_sq, offset);
        }

        if (tid == 0) {
            warp_sums[0] = final_sum;
            warp_sums[1] = rsqrt((final_sum_sq / normalized_size) - 
                                (final_sum * final_sum / (normalized_size * normalized_size)) + 
                                static_cast<accscalar_t>(eps));
            warp_sums[2] = final_sum / normalized_size;
        }
    }
    __syncthreads();

    const accscalar_t mean = warp_sums[2];
    const accscalar_t inv_std = warp_sums[1];

    #pragma unroll 4
    for (int v = 0; v < vectors_per_thread; v++) {
        const int idx = tid * vector_size + v * blockDim.x * vector_size;
        if (idx < normalized_size - (vector_size - 1)) {
            accscalar_t vals[vector_size];
            accscalar_t weights[vector_size];
            accscalar_t biases[vector_size];

            #pragma unroll
            for (int i = 0; i < vector_size; i++) {
                vals[i] = static_cast<accscalar_t>(in_ptr[idx + i]);
                weights[i] = static_cast<accscalar_t>(weight[idx + i]);
                biases[i] = static_cast<accscalar_t>(bias[idx + i]);
                
                vals[i] = (vals[i] - mean) * inv_std;
                out_ptr[idx + i] = static_cast<scalar_t>(vals[i] * weights[i] + biases[i]);
            }
        }
    }

    if (remainder_start + tid < normalized_size) {
        const int idx = remainder_start + tid;
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        val = (val - mean) * inv_std;
        out_ptr[idx] = static_cast<scalar_t>(
            val * static_cast<accscalar_t>(weight[idx]) + 
            static_cast<accscalar_t>(bias[idx]));
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    
    const int normalized_size = weight.numel();
    const int outer_size = x.numel() / normalized_size;
    
    const int threads = 256;
    const dim3 blocks(outer_size);
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        const int warps_per_block = threads >> 5;
        const int shared_mem_size = warps_per_block * 3 * sizeof(accscalar_t);
        
        layernorm_uniform_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
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