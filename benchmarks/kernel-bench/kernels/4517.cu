#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Kernel using stride loops to handle larger workloads efficiently.
template <typename scalar_t>
__global__ void compute_stats_kernel_stride(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var) {

    const int tid = threadIdx.x;
    const int n = blockIdx.y;
    const int g = blockIdx.z;
    const int group_size = channels_per_group * spatial;
    const int group_offset = n * C * spatial + g * channels_per_group * spatial;

    __shared__ scalar_t s_sum[256];
    __shared__ scalar_t s_sum_sq[256];

    scalar_t sum = 0;
    scalar_t sum_sq = 0;

    for (int i = tid; i < group_size; i += blockDim.x) {
        const int global_idx = group_offset + i;
        const scalar_t val = x[global_idx];
        sum += val;
        sum_sq += val * val;
    }

    s_sum[tid] = sum;
    s_sum_sq[tid] = sum_sq;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
            s_sum_sq[tid] += s_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        const scalar_t group_mean = s_sum[0] / group_size;
        const scalar_t group_var = s_sum_sq[0] / group_size - group_mean * group_mean;
        const int out_idx = n * num_groups + g;
        mean[out_idx] = group_mean;
        var[out_idx] = group_var;
    }
}

template <typename scalar_t>
__global__ void group_norm_forward_kernel_stride(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = N * C * spatial;

    for (int idx = tid; idx < total; idx += stride) {
        const int j = idx % spatial;
        const int temp = idx / spatial;
        const int c = temp % C;
        const int n = temp / C;

        const int g = c / channels_per_group;
        const int stats_idx = n * num_groups + g;

        const scalar_t m = mean[stats_idx];
        const scalar_t v = var[stats_idx];
        const scalar_t inv_std = rsqrt(v + eps);
        const scalar_t x_val = x[idx];
        y[idx] = (x_val - m) * inv_std * weight[c] + bias[c];
    }
}

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

    const int N = x.size(0);
    const int C = x.size(1);
    int spatial = 1;
    for (int i = 2; i < x.dim(); i++) {
        spatial *= x.size(i);
    }

    const int channels_per_group = C / num_groups;
    const int group_size = channels_per_group * spatial;
    const int block_size = 256;

    auto y = torch::empty_like(x);
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto mean = torch::empty({N, num_groups}, options);
    auto var = torch::empty({N, num_groups}, options);

    dim3 stats_blocks(1, N, num_groups);
    dim3 norm_blocks((N * C * spatial + block_size - 1) / block_size);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
        compute_stats_kernel_stride<scalar_t><<<stats_blocks, block_size, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            N, C, spatial,
            channels_per_group,
            num_groups,
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>());

        group_norm_forward_kernel_stride<scalar_t><<<norm_blocks, block_size, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            N, C, spatial,
            channels_per_group,
            num_groups,
            static_cast<scalar_t>(eps),
            y.data_ptr<scalar_t>());
    }));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA)");
}