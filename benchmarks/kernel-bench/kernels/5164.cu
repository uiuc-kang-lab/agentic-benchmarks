#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_vectorized_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

    using accscalar_t = at::acc_type<scalar_t, true>;
    constexpr int VEC_SIZE = 4;
    using load_t = typename std::conditional<VEC_SIZE == 4 && sizeof(scalar_t) == 4, float4, scalar_t>::type;

    int instance_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;

    extern __shared__ char smem[];
    accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* s_sum_sq = s_sum + blockDim.x;

    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;

    // Vectorized accumulation
    const int vec_limit = normalized_size - (normalized_size % VEC_SIZE);
    for (int i = tid * VEC_SIZE; i < vec_limit; i += blockDim.x * VEC_SIZE) {
        load_t vec_data = *reinterpret_cast<const load_t*>(in_ptr + i);
        scalar_t* elements = reinterpret_cast<scalar_t*>(&vec_data);
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            accscalar_t val = static_cast<accscalar_t>(elements[j]);
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    // Handle remaining elements
    for (int i = vec_limit + tid; i < normalized_size; i += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    if ((tid & 31) == 0) {
        s_sum[tid >> 5] = local_sum;
        s_sum_sq[tid >> 5] = local_sum_sq;
    }
    __syncthreads();

    // Block-level reduction
    if (tid < 32) {
        accscalar_t block_sum = (tid < (blockDim.x >> 5)) ? s_sum[tid] : 0;
        accscalar_t block_sum_sq = (tid < (blockDim.x >> 5)) ? s_sum_sq[tid] : 0;
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
            block_sum_sq += __shfl_down_sync(0xffffffff, block_sum_sq, offset);
        }

        if (tid == 0) {
            accscalar_t mean = block_sum / normalized_size;
            accscalar_t var = block_sum_sq / normalized_size - mean * mean;
            s_sum[0] = mean;
            s_sum[1] = rsqrt(var + static_cast<accscalar_t>(eps));
        }
    }
    __syncthreads();

    const accscalar_t mean = s_sum[0];
    const accscalar_t inv_std = s_sum[1];

    // Vectorized normalization
    for (int i = tid * VEC_SIZE; i < vec_limit; i += blockDim.x * VEC_SIZE) {
        load_t vec_data = *reinterpret_cast<const load_t*>(in_ptr + i);
        scalar_t* in_elements = reinterpret_cast<scalar_t*>(&vec_data);
        scalar_t out_elements[VEC_SIZE];

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            accscalar_t val = static_cast<accscalar_t>(in_elements[j]);
            val = (val - mean) * inv_std;
            out_elements[j] = static_cast<scalar_t>(val * weight[i + j] + bias[i + j]);
        }

        *reinterpret_cast<load_t*>(out_ptr + i) = *reinterpret_cast<load_t*>(out_elements);
    }

    // Handle remaining elements
    for (int i = vec_limit + tid; i < normalized_size; i += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
        val = (val - mean) * inv_std;
        out_ptr[i] = static_cast<scalar_t>(val * weight[i] + bias[i]);
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    int normalized_size = weight.numel();
    int outer_size = x.numel() / normalized_size;

    int threads = std::min(1024, (normalized_size + 3) / 4);
    threads = (threads + 31) & ~31;
    int blocks = outer_size;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        int shared_size = (threads / 32) * 2 * sizeof(accscalar_t);
        layernorm_vectorized_kernel<scalar_t><<<blocks, threads, shared_size>>>(
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