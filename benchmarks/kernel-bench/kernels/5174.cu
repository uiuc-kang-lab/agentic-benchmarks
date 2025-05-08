#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

// Optimized LayerNorm kernel with adjustable block size.

template <typename scalar_t>
__global__ void layernorm_optimized_block_size_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int outer_size) {

    using accscalar_t = at::acc_type<scalar_t, true>;
    
    // Adjust block size as needed
    const int tid = threadIdx.x;
    const int instance_idx = blockIdx.x;
    
    extern __shared__ accscalar_t smem[2 * blockDim.x];
    accscalar_t* s_sum = smem;
    accscalar_t* s_sum_sq = s_sum + blockDim.x;
    
    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;
    
    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;
    
    // Loop over elements in a strided manner
    for (int i = tid; i < normalized_size; i += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();
    
    // Reduce sums within the block
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
    
    // Compute mean and variance
    if (tid == 0) {
        accscalar_t mean = s_sum[0] / static_cast<accscalar_t>(normalized_size);
        accscalar_t variance = s_sum_sq[0] / static_cast<accscalar_t>(normalized_size) - mean * mean;
        s_sum[0] = mean;
        s_sum_sq[0] = rsqrt(variance + static_cast<accscalar_t>(eps));
    }
    __syncthreads();
    
    accscalar_t mean = s_sum[0];
    accscalar_t inv_std = s_sum_sq[0];
    
    // Normalize and apply affine transformation
    for (int i = tid; i < normalized_size; i += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
        accscalar_t norm_val = (val - mean) * inv_std;
        out_ptr[i] = static_cast<scalar_t>(norm_val * static_cast<accscalar_t>(weight[i]) + static_cast<accscalar_t>(bias[i]));
    }
}

// Host function to dispatch the kernel with variable block size

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    const int normalized_size = weight.numel();
    const int outer_size = x.numel() / normalized_size;

    // Experiment with different block sizes
    const int block_size = 256; // Example of an optimized block size
    const int blocks = outer_size;
    const int shared_mem_size = block_size * 2 * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        layernorm_optimized_block_size_kernel<scalar_t><<<blocks, block_size, shared_mem_size>>>(
            x.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            static_cast<float>(eps),
            output.data_ptr<scalar_t>(),
            normalized_size,
            outer_size);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}
