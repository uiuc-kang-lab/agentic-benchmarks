#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <ATen/AccumulateType.h>

__global__ void layernorm_strided_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float eps,
    float* __restrict__ output,
    const int normalized_size,
    const int outer_size) {

    using accscalar_t = at::acc_type<float, true>;
    const int tid = threadIdx.x;
    const int instance_idx = blockIdx.x;

    extern __shared__ accscalar_t shared_mem[];
    accscalar_t* shared_sum = shared_mem;
    accscalar_t* shared_sum_sq = shared_mem + blockDim.x;

    const float* in_ptr = input + instance_idx * normalized_size;
    float* out_ptr = output + instance_idx * normalized_size;

    accscalar_t local_sum = 0;
    accscalar_t local_sum_sq = 0;

    for (int idx = tid; idx < normalized_size; idx += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        local_sum += val;
        local_sum_sq += val * val;
    }

    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
            shared_sum_sq[tid] += shared_sum_sq[tid + offset];
        }
        __syncthreads();
    }

    __shared__ accscalar_t mean, inv_std;
    if (tid == 0) {
        mean = shared_sum[0] / normalized_size;
        accscalar_t variance = shared_sum_sq[0] / normalized_size - mean * mean;
        inv_std = rsqrtf(variance + static_cast<accscalar_t>(eps));
    }
    __syncthreads();

    for (int idx = tid; idx < normalized_size; idx += blockDim.x) {
        accscalar_t val = static_cast<accscalar_t>(in_ptr[idx]);
        accscalar_t norm_val = (val - mean) * inv_std;
        out_ptr[idx] = static_cast<float>(norm_val * weight[idx] + bias[idx]);
    }
}

void layernorm_forward_strided(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    torch::Tensor& output,
    double eps) {

    const int normalized_size = weight.numel();
    const int outer_size = x.numel() / normalized_size;

    const int threads = 1024;
    const dim3 blocks(outer_size);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "layernorm_forward_strided_cuda", ([&] {
        layernorm_strided_kernel<<<blocks, threads, threads * 2 * sizeof(float)>>>(
            x.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            static_cast<float>(eps),
            output.data_ptr<float>(),
            normalized_size,
            outer_size);
    }));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_strided", &layernorm_forward_strided, "LayerNorm forward with stride loops (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias"), py::arg("output"), py::arg("eps") = 1e-5);
}