#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_adaptive_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

    const int instance_idx = blockIdx.x;
    const int tid = threadIdx.x;

    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;

    using accscalar_t = at::acc_type<scalar_t, true>;
    extern __shared__ __align__(sizeof(accscalar_t)) char smem[];
    accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* s_sum_sq = s_sum + blockDim.x;

    // Vectorized loading (4 elements per thread)
    const int vec_size = 4;
    accscalar_t local_sum[vec_size] = {0};
    accscalar_t local_sum_sq[vec_size] = {0};
    const int num_vectors = normalized_size / (blockDim.x * vec_size);

    #pragma unroll
    for (int i = 0; i < num_vectors; ++i) {
        int idx = i * blockDim.x * vec_size + tid * vec_size;
        scalar_t temp[vec_size];
        *(float4*)(&temp) = *(float4*)(in_ptr + idx);

        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
            accscalar_t val = static_cast<accscalar_t>(temp[j]);
            local_sum[j] += val;
            local_sum_sq[j] += val * val;
        }
    }

    // Final elements
    for (int idx = num_vectors * blockDim.x * vec_size + tid; 
         idx < normalized_size; 
         idx += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        local_sum[0] += val;
        local_sum_sq[0] += val * val;
    }

    // Warp-level reduction
    for (int j = 0; j < vec_size; j++) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum[j] += __shfl_down_sync(0xffffffff, local_sum[j], offset);
            local_sum_sq[j] += __shfl_down_sync(0xffffffff, local_sum_sq[j], offset);
        }
    }

    if (tid == 0) {
        s_sum[0] = local_sum[0];
        s_sum_sq[0] = local_sum_sq[0];
        for (int j = 1; j < vec_size; j++) {
            s_sum[0] += local_sum[j];
            s_sum_sq[0] += local_sum_sq[j];
        }
    }
    __syncthreads();

    // Compute mean and std
    const accscalar_t mean = s_sum[0] / normalized_size;
    const accscalar_t inv_std = rsqrt(s_sum_sq[0]/normalized_size - mean*mean + eps);

    // Vectorized writing with transformed values
    #pragma unroll
    for (int i = 0; i < num_vectors; ++i) {
        int idx = i * blockDim.x * vec_size + tid * vec_size;
        scalar_t temp[vec_size];
        *(float4*)(&temp) = *(float4*)(in_ptr + idx);

        scalar_t result[vec_size];
        #pragma unroll
        for (int j = 0; j < vec_size; j++) {
            accscalar_t val = static_cast<accscalar_t>(temp[j]);
            result[j] = static_cast<scalar_t>(((val - mean) * inv_std) * 
                        static_cast<accscalar_t>(weight[idx + j]) + 
                        static_cast<accscalar_t>(bias[idx + j]));
        }
        *(float4*)(out_ptr + idx) = *(float4*)(&result);
    }

    // Final elements
    for (int idx = num_vectors * blockDim.x * vec_size + tid; 
         idx < normalized_size; 
         idx += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        out_ptr[idx] = static_cast<scalar_t>(((val - mean) * inv_std) * 
                        static_cast<accscalar_t>(weight[idx]) + 
                        static_cast<accscalar_t>(bias[idx]));
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    int normalized_size = weight.numel();
    int outer_size = x.numel() / normalized_size;

    // Choose block size based on normalized_size (multiple of 32 up to 1024)
    int block_size;
    if (normalized_size <= 256) {
        block_size = 256;
    } else if (normalized_size <= 512) {
        block_size = 512;
    } else if (normalized_size <= 1024) {
        block_size = 1024;
    } else {
        block_size = 128; 
    }
    block_size = std::min(block_size, (normalized_size + 31) / 32 * 32);
    block_size = std::min(1024, block_size);

    dim3 grid(outer_size);
    dim3 block(block_size);

    const size_t shared_size = 2 * block_size * sizeof(at::acc_type<float, true>);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        layernorm_adaptive_kernel<scalar_t><<<grid, block, shared_size>>>(
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