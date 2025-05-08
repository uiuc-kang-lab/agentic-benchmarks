#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

template <typename scalar_t>
__global__ void layernorm_shared_input_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const float eps,
    scalar_t* __restrict__ output,
    const int normalized_size,
    const int outer_size) {

    using accscalar_t = at::acc_type<scalar_t, true>;
    
    extern __shared__ char smem[];
    scalar_t* s_input = reinterpret_cast<scalar_t*>(smem);
    accscalar_t* s_reduce = reinterpret_cast<accscalar_t*>(s_input + normalized_size);
    
    const int tidx = threadIdx.x;
    const int tidy = threadIdx.y;
    const int instance_idx = blockIdx.x;
    const int thread_id = tidy * blockDim.x + tidx;
    const int thread_stride = blockDim.x * blockDim.y;

    // Load input into shared memory
    const scalar_t* in_ptr = input + instance_idx * normalized_size;
    for (int idx = thread_id; idx < normalized_size; idx += thread_stride) {
        s_input[idx] = in_ptr[idx];
    }
    __syncthreads();

    // First-stage reduction with warp shuffles
    accscalar_t sum = 0, sum_sq = 0;
    for (int idx = thread_id; idx < normalized_size; idx += thread_stride) {
        accscalar_t val = static_cast<accscalar_t>(s_input[idx]);
        sum += val;
        sum_sq += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }

    // Store warp results in shared memory
    if (tidx % 32 == 0) {
        s_reduce[(tidy * 32) + (tidx/32)] = sum;
        s_reduce[(tidy * 32) + (tidx/32) + blockDim.x*blockDim.y/32] = sum_sq;
    }
    __syncthreads();

    // Final reduction by first warp
    if (thread_id < 32) {
        accscalar_t final_sum = 0, final_sum_sq = 0;
        for (int i = thread_id; i < blockDim.x*blockDim.y/32; i += 32) {
            final_sum += s_reduce[i];
            final_sum_sq += s_reduce[i + blockDim.x*blockDim.y/32];
        }
        
        for (int offset = 16; offset > 0; offset >>= 1) {
            final_sum += __shfl_down_sync(0xffffffff, final_sum, offset);
            final_sum_sq += __shfl_down_sync(0xffffffff, final_sum_sq, offset);
        }

        if (thread_id == 0) {
            s_reduce[0] = final_sum / normalized_size;
            accscalar_t var = (final_sum_sq / normalized_size) - (s_reduce[0] * s_reduce[0]);
            s_reduce[1] = rsqrt(var + static_cast<accscalar_t>(eps));
        }
    }
    __syncthreads();

    // Normalize using shared memory input
    scalar_t* out_ptr = output + instance_idx * normalized_size;
    for (int idx = thread_id; idx < normalized_size; idx += thread_stride) {
        accscalar_t val = static_cast<accscalar_t>(s_input[idx]);
        accscalar_t normalized = (val - s_reduce[0]) * s_reduce[1];
        out_ptr[idx] = static_cast<scalar_t>(
            normalized * static_cast<accscalar_t>(weight[idx]) + 
            static_cast<accscalar_t>(bias[idx]));
    }
}

torch::Tensor layernorm_forward(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps = 1e-5) {
    auto output = torch::empty_like(x);
    const int normalized_size = weight.numel();
    const int outer_size = x.numel() / normalized_size;

    dim3 threads(32, 32);
    int shared_mem = normalized_size * sizeof(typename torch::ScalarType<at::kHalf>::underlying) + 
                    2 * threads.x * threads.y * sizeof(float);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_cuda", ([&] {
        layernorm_shared_input_kernel<scalar_t><<<outer_size, threads, shared_mem>>>(
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