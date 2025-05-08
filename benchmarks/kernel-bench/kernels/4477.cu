#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

// Optimized warp reduction with no divergent branches
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized compute_stats_kernel with improved thread/block indexing
// Using 2D grid: blockIdx.y for samples, blockIdx.x for groups
// Each block computes the mean and variance for one (n, g) group

template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var) {

    const int n = blockIdx.y;
    const int g = blockIdx.x;
    const int group_offset = n * C * spatial + g * channels_per_group * spatial;
    const int group_elems = channels_per_group * spatial;

    scalar_t thread_sum = 0;
    scalar_t thread_sum_sq = 0;

    for (int i = threadIdx.x; i < group_elems; i += blockDim.x) {
        const int c = i / spatial;
        const int j = i % spatial;
        const scalar_t val = x[group_offset + c * spatial + j];
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    thread_sum = warpReduceSum(thread_sum);
    thread_sum_sq = warpReduceSum(thread_sum_sq);

    __shared__ scalar_t s_partial_sums[32];
    __shared__ scalar_t s_partial_squares[32];

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    if (lane_id == 0) {
        s_partial_sums[warp_id] = thread_sum;
        s_partial_squares[warp_id] = thread_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < (blockDim.x + warpSize - 1) / warpSize) {
        thread_sum = s_partial_sums[lane_id];
        thread_sum_sq = s_partial_squares[lane_id];

        thread_sum = warpReduceSum(thread_sum);
        thread_sum_sq = warpReduceSum(thread_sum_sq);

        if (lane_id == 0) {
            const scalar_t group_mean = thread_sum / group_elems;
            const scalar_t group_var = thread_sum_sq / group_elems - group_mean * group_mean;
            const int out_index = n * num_groups + g;
            mean[out_index] = group_mean;
            var[out_index] = group_var;
        }
    }
}

// Group Norm forward kernel using 2D grid indexing
// grid.y covers the (n, c) pairs
// grid.x covers the spatial dimension in tiles

template <typename scalar_t>
__global__ void group_norm_forward_kernel(
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

    const int nc = blockIdx.y * blockDim.y + threadIdx.y;
    if (nc >= N * C) return;
    const int n = nc / C;
    const int c = nc % C;

    const int g = c / channels_per_group;
    const int stats_index = n * num_groups + g;
    const scalar_t m = mean[stats_index];
    const scalar_t v = var[stats_index];
    const scalar_t inv_std = rsqrt(v + eps);

    const int tile_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (tile_index < spatial) {
        const int index = (n * C + c) * spatial + tile_index;
        const scalar_t x_val = x[index];
        y[index] = (x_val - m) * inv_std * weight[c] + bias[c];
    }
}

// Host function to launch the kernels with improved thread/block indexing
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

    auto y = torch::empty_like(x);
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto mean = torch::empty({N, num_groups}, options);
    auto var = torch::empty({N, num_groups}, options);

    dim3 threads_stats(256);
    dim3 blocks_stats(num_groups, N);

    dim3 threads_norm(32, 8); // 32 threads for spatial, 8 for (N, C) pairs
    dim3 blocks_norm((spatial + threads_norm.x - 1) / threads_norm.x, (N * C + threads_norm.y - 1) / threads_norm.y);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
        compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            N,
            C,
            spatial,
            channels_per_group,
            num_groups,
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>());

        group_norm_forward_kernel<scalar_t><<<blocks_norm, threads_norm, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            N,
            C,
            spatial,
            channels_per_group,
            num_groups,
            static_cast<scalar_t>(eps),
            y.data_ptr<scalar_t>());
    }));

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with improved thread/block indexing");
}
