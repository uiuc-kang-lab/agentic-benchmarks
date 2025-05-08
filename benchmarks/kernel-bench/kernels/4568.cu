#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define BLOCK_SIZE_STATS 256
#define BLOCK_SIZE_NORM 256

template<typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void compute_stats_kernel_warp(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var) {

    const int idx = blockIdx.x;
    const int n = idx / num_groups;
    const int g = idx % num_groups;
    
    const int group_offset = n * C * spatial + g * channels_per_group * spatial;
    const int group_elems = channels_per_group * spatial;

    scalar_t sum = 0;
    scalar_t sum_sq = 0;
    for (int i = threadIdx.x; i < group_elems; i += BLOCK_SIZE) {
        int c = i / spatial;
        int j = i % spatial;
        scalar_t val = x[group_offset + c * spatial + j];
        sum += val;
        sum_sq += val * val;
    }

    // Warp-level reduction
    sum = warpReduceSum(sum);
    sum_sq = warpReduceSum(sum_sq);

    extern __shared__ char smem[];
    scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
    scalar_t* s_sum_sq = s_sum + (BLOCK_SIZE/32);

    if (threadIdx.x % 32 == 0) {
        s_sum[threadIdx.x/32] = sum;
        s_sum_sq[threadIdx.x/32] = sum_sq;
    }
    __syncthreads();

    if (threadIdx.x < 32) {
        scalar_t block_sum = (threadIdx.x < blockDim.x/32) ? s_sum[threadIdx.x] : 0;
        scalar_t block_sum_sq = (threadIdx.x < blockDim.x/32) ? s_sum_sq[threadIdx.x] : 0;
        
        block_sum = warpReduceSum(block_sum);
        block_sum_sq = warpReduceSum(block_sum_sq);

        if (threadIdx.x == 0) {
            scalar_t group_mean = block_sum / group_elems;
            scalar_t group_var = block_sum_sq / group_elems - group_mean * group_mean;
            int out_index = n * num_groups + g;
            mean[out_index] = group_mean;
            var[out_index] = group_var;
        }
    }
}

template <typename scalar_t, int BLOCK_SIZE>
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

    int index = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int total = N * C * spatial;
    if (index >= total) return;

    int j = index % spatial;
    int temp = index / spatial;
    int c = temp % C;
    int n = temp / C;
    
    int g = c / channels_per_group;
    int stats_index = n * num_groups + g;
    scalar_t m = mean[stats_index];
    scalar_t v = var[stats_index];
    scalar_t inv_std = rsqrt(v + eps);
    scalar_t x_val = x[index];
    y[index] = (x_val - m) * inv_std * weight[c] + bias[c];
}

torch::Tensor group_norm_forward_warp(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

    const int N = x.size(0);
    const int C = x.size(1);
    int spatial = 1;
    for (int i = 2; i < x.dim(); i++)
        spatial *= x.size(i);
    const int channels_per_group = C / num_groups;

    auto y = torch::empty_like(x);
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto mean = torch::empty({N, num_groups}, options);
    auto var = torch::empty({N, num_groups}, options);

    const int total_groups = N * num_groups;
    const int group_elems = channels_per_group * spatial;
    const int threads_stats = BLOCK_SIZE_STATS;
    const int total_elements = N * C * spatial;
    const int blocks_norm = (total_elements + BLOCK_SIZE_NORM - 1) / BLOCK_SIZE_NORM;

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_warp", ([&] {
        size_t shared_mem = (BLOCK_SIZE_STATS/32)*2*sizeof(scalar_t);
        compute_stats_kernel_warp<scalar_t, BLOCK_SIZE_STATS><<<
            total_groups, BLOCK_SIZE_STATS, shared_mem, stream>>>(
                x.data_ptr<scalar_t>(),
                N, C, spatial,
                channels_per_group,
                num_groups,
                mean.data_ptr<scalar_t>(),
                var.data_ptr<scalar_t>());

        group_norm_forward_kernel<scalar_t, BLOCK_SIZE_NORM><<<
            blocks_norm, BLOCK_SIZE_NORM, 0, stream>>>(
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
    m.def("forward", &group_norm_forward_warp, "GroupNorm forward with warp reductions (CUDA)");
}
