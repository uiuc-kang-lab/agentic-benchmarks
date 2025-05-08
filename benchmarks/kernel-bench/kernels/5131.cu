#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

template <typename scalar_t>
__global__ void layernorm_hybrid_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size) {

    int instance_idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    scalar_t* out_ptr = output + instance_idx * normalized_size;

    using accscalar_t = at::acc_type<scalar_t, true>;

    extern __shared__ char smem[];
    int num_warps = blockDim.x / 32;
    accscalar_t* s_sum = reinterpret_cast<accscalar_t*>(smem);
    accscalar_t* s_sum_sq = s_sum + num_warps;

    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;
    
    const int CHUNK_SIZE = 4;
    const int num_chunks = (normalized_size + blockDim.x * CHUNK_SIZE - 1) / (blockDim.x * CHUNK_SIZE);
    
    #pragma unroll
    for (int chunk = 0; chunk < num_chunks; chunk++) {
        const int chunk_start = chunk * blockDim.x * CHUNK_SIZE + tid;
        
        #pragma unroll
        for (int i = 0; i < CHUNK_SIZE; i++) {
            const int idx = chunk_start + i * blockDim.x;
            if (idx < normalized_size) {
                accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
                local_sum += val;
                local_sum_sq += val * val;
            }
        }
    }

    accscalar_t warp_sum = warpReduceSum(local_sum);
    accscalar_t warp_sum_sq = warpReduceSum(local_sum_sq);

    if (lane_id == 0) {
        s_sum[warp_id] = warp_sum;
        s_sum_sq[warp_id] = warp_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        accscalar_t block_sum = (lane_id < num_warps) ? s_sum[lane_id] : 0;
        accscalar_t block_sum_sq = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0;

        accscalar_t total_sum = warpReduceSum(block_sum);
        accscalar_t total_sum_sq = warpReduceSum(block_sum_sq);

        if (lane_id == 0) {
            s_sum[0] = total_sum / normalized_size;  // mean
            s_sum_sq[0] = rsqrtf(total_sum_sq / normalized_size - s_sum[0] * s_sum[0] + eps);  // inv_std
        }
    }
    __syncthreads();

    accscalar_t mean = s_sum[0];
    accscalar_t inv_std = s_sum_sq[0];
    
    for (int idx = tid; idx < normalized_size; idx += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        accscalar_t norm_val = (val - mean) * inv_std;
        out_ptr[idx] = static_cast<scalar_t>(norm_val * weight[idx] + bias[idx]);
    }
}

torch::Tensor layernorm_forward_hybrid(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    int normalized_size = weight.numel();
    int outer_size = x.numel() / normalized_size;

    int threads = std::min(((normalized_size + 127) / 128) * 32, 1024);
    threads = ((threads + 31) / 32) * 32;  // Ensure multiple of warp size
    int num_warps = threads / 32;
    int shared_size = 2 * num_warps * sizeof(at::acc_type<float, true>);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_hybrid", ([&] {
        layernorm_hybrid_kernel<scalar_t><<<outer_size, threads, shared_size>>>(
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
    m.def("forward", &layernorm_forward_hybrid, "LayerNorm forward hybrid (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("eps") = 1e-5);
}