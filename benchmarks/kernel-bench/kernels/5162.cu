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

    const int instance_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid % warpSize;
    const int warp_id = tid / warpSize;

    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;

    using accscalar_t = at::acc_type<scalar_t, true>;
    using vec_t = typename std::conditional<std::is_same<scalar_t, float>::value, float4, double2>::type;
    constexpr int VEC_SIZE = std::is_same<scalar_t, float>::value ? 4 : 2;

    extern __shared__ char smem[];
    accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* s_sum_sq = s_sum + blockDim.x / warpSize;

    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;

    // Vectorized load and accumulation
    const int vec_limit = normalized_size / VEC_SIZE;
    for (int i = tid; i < vec_limit; i += blockDim.x) {
        vec_t vec_in;
        *reinterpret_cast<vec_t*>(&vec_in) = *reinterpret_cast<const vec_t*>(&in_ptr[i * VEC_SIZE]);
        
        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            accscalar_t val = reinterpret_cast<scalar_t*>(&vec_in)[j];
            local_sum += val;
            local_sum_sq += val * val;
        }
    }

    // Handle remaining elements
    for (int i = vec_limit * VEC_SIZE + tid; i < normalized_size; i += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xffffffff, local_sum_sq, offset);
    }

    // Store warp-level partial sums in shared memory
    if (lane == 0) {
        atomicAdd(&s_sum[warp_id], local_sum);
        atomicAdd(&s_sum_sq[warp_id], local_sum_sq);
    }
    __syncthreads();

    // Final reduction by first warp
    if (tid < warpSize) {
        accscalar_t warp_sum = tid < (blockDim.x / warpSize) ? s_sum[tid] : 0;
        accscalar_t warp_sum_sq = tid < (blockDim.x / warpSize) ? s_sum_sq[tid] : 0;

        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            warp_sum += __shfl_down_sync(0xffffffff, warp_sum, offset);
            warp_sum_sq += __shfl_down_sync(0xffffffff, warp_sum_sq, offset);
        }

        if (tid == 0) {
            accscalar_t mean = warp_sum / normalized_size;
            accscalar_t var = warp_sum_sq / normalized_size - mean * mean;
            s_sum[0] = mean;
            s_sum_sq[0] = rsqrt(var + static_cast<accscalar_t>(eps));
        }
    }
    __syncthreads();

    const accscalar_t mean = s_sum[0];
    const accscalar_t inv_std = s_sum_sq[0];

    // Vectorized store with normalization
    for (int i = tid; i < vec_limit; i += blockDim.x) {
        vec_t vec_in, vec_weight, vec_bias;
        *reinterpret_cast<vec_t*>(&vec_in) = *reinterpret_cast<const vec_t*>(&in_ptr[i * VEC_SIZE]);
        *reinterpret_cast<vec_t*>(&vec_weight) = *reinterpret_cast<const vec_t*>(&weight[i * VEC_SIZE]);
        *reinterpret_cast<vec_t*>(&vec_bias) = *reinterpret_cast<const vec_t*>(&bias[i * VEC_SIZE]);

        #pragma unroll
        for (int j = 0; j < VEC_SIZE; j++) {
            accscalar_t val = reinterpret_cast<scalar_t*>(&vec_in)[j];
            val = (val - mean) * inv_std;
            val = val * reinterpret_cast<scalar_t*>(&vec_weight)[j] + reinterpret_cast<scalar_t*>(&vec_bias)[j];
            reinterpret_cast<scalar_t*>(&vec_in)[j] = val;
        }

        *reinterpret_cast<vec_t*>(&out_ptr[i * VEC_SIZE]) = vec_in;
    }

    // Handle remaining elements
    for (int i = vec_limit * VEC_SIZE + tid; i < normalized_size; i += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[i]);
        val = (val - mean) * inv_std;
        out_ptr[i] = static_cast<scalar_t>(val * weight[i] + bias[i]);
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    const int normalized_size = weight.numel();
    const int outer_size = x.numel() / normalized_size;

    constexpr int threads = 256;
    const int shared_size = (threads / 32) * 2 * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        layernorm_vectorized_kernel<scalar_t><<<outer_size, threads, shared_size>>>(
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