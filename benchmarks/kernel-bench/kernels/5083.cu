#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ void compute_partial_sums(
    const scalar_t* __restrict__ in_ptr,
    accscalar_t& local_sum,
    accscalar_t& local_sum_sq,
    const int tid,
    const int normalized_size,
    const int stride) {
    
    local_sum = 0;
    local_sum_sq = 0;
    for (int i = tid; i < normalized_size; i += stride) {
        accscalar_t val = static_cast<accscalar_t>(__ldg(&in_ptr[i]));
        local_sum += val;
        local_sum_sq += val * val;
    }
}

template <typename accscalar_t>
__device__ __forceinline__ void reduce_partial_sums(
    accscalar_t* s_sum,
    accscalar_t* s_sum_sq,
    const int tid,
    const int block_size) {
    
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }
}

template <typename accscalar_t>
__device__ __forceinline__ void compute_stats(
    const accscalar_t sum,
    const accscalar_t sum_sq,
    const int normalized_size,
    const float eps,
    accscalar_t& mean,
    accscalar_t& inv_std) {
    
    mean = sum / static_cast<accscalar_t>(normalized_size);
    accscalar_t var = sum_sq / static_cast<accscalar_t>(normalized_size) - mean * mean;
    inv_std = rsqrt(var + static_cast<accscalar_t>(eps));
}

template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ void normalize_and_scale(
    const scalar_t* __restrict__ in_ptr,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    scalar_t* __restrict__ out_ptr,
    const accscalar_t mean,
    const accscalar_t inv_std,
    const int tid,
    const int normalized_size,
    const int stride) {
    
    for (int i = tid; i < normalized_size; i += stride) {
        accscalar_t val = static_cast<accscalar_t>(__ldg(&in_ptr[i]));
        accscalar_t norm_val = (val - mean) * inv_std;
        accscalar_t w = static_cast<accscalar_t>(__ldg(&weight[i]));
        accscalar_t b = static_cast<accscalar_t>(__ldg(&bias[i]));
        out_ptr[i] = static_cast<scalar_t>(norm_val * w + b);
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

    using accscalar_t = at::acc_type<scalar_t, true>;
    
    const int tid = threadIdx.x;
    const int instance_idx = blockIdx.x;
    const int stride = blockDim.x;
    
    // Get pointers to current instance
    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;

    // Shared memory allocation
    extern __shared__ char smem[];
    accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* s_sum_sq = s_sum + blockDim.x;

    // Step 1: Compute partial sums
    accscalar_t local_sum, local_sum_sq;
    compute_partial_sums<scalar_t, accscalar_t>(
        in_ptr, local_sum, local_sum_sq, tid, normalized_size, stride);
    
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Step 2: Reduce partial sums
    reduce_partial_sums<accscalar_t>(s_sum, s_sum_sq, tid, blockDim.x);

    // Step 3: Compute mean and inverse standard deviation
    __shared__ accscalar_t mean, inv_std;
    if (tid == 0) {
        compute_stats<accscalar_t>(
            s_sum[0], s_sum_sq[0], normalized_size, eps, mean, inv_std);
    }
    __syncthreads();

    // Step 4: Normalize and apply affine transformation
    normalize_and_scale<scalar_t, accscalar_t>(
        in_ptr, weight, bias, out_ptr, mean, inv_std, tid, normalized_size, stride);
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    
    const int normalized_size = weight.numel();
    const int outer_size = x.numel() / normalized_size;
    const int threads = std::min(normalized_size, 1024);
    const int blocks = outer_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        using accscalar_t = at::acc_type<scalar_t, true>;
        const int shared_size = threads * 2 * sizeof(accscalar_t);
        
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
    m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA) with modular design",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}