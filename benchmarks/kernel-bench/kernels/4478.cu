#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

template <typename scalar_t>
__device__ __forceinline__ scalar_t warpReduceSum(scalar_t val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

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

    const int group_idx = blockIdx.x;
    const int n = group_idx / num_groups;
    const int g = group_idx % num_groups;
    
    const int group_size = channels_per_group * spatial;
    const int group_offset = n * C * spatial + g * channels_per_group * spatial;
    
    // Each thread handles multiple elements with grid-stride loop
    scalar_t thread_sum = 0;
    scalar_t thread_sum_sq = 0;

    // Stride loop to handle large workloads
    const int num_elements_per_thread = (group_size + blockDim.x - 1) / blockDim.x;
    const int thread_start = threadIdx.x * num_elements_per_thread;
    const int thread_end = min(thread_start + num_elements_per_thread, group_size);

    #pragma unroll 2
    for (int idx = thread_start; idx < thread_end; idx++) {
        const int c = idx / spatial;
        const int s = idx % spatial;
        const scalar_t val = x[group_offset + c * spatial + s];
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    // Warp-level reduction
    thread_sum = warpReduceSum(thread_sum);
    thread_sum_sq = warpReduceSum(thread_sum_sq);

    __shared__ scalar_t s_sum[32];
    __shared__ scalar_t s_sum_sq[32];

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    if (lane_id == 0) {
        s_sum[warp_id] = thread_sum;
        s_sum_sq[warp_id] = thread_sum_sq;
    }
    __syncthreads();

    // Final reduction in first warp
    if (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) {
        thread_sum = threadIdx.x < blockDim.x / warpSize ? s_sum[threadIdx.x] : 0;
        thread_sum_sq = threadIdx.x < blockDim.x / warpSize ? s_sum_sq[threadIdx.x] : 0;
        
        thread_sum = warpReduceSum(thread_sum);
        thread_sum_sq = warpReduceSum(thread_sum_sq);

        if (threadIdx.x == 0) {
            const scalar_t inv_N = scalar_t(1.0) / group_size;
            const scalar_t group_mean = thread_sum * inv_N;
            const scalar_t group_var = thread_sum_sq * inv_N - group_mean * group_mean;
            mean[n * num_groups + g] = group_mean;
            var[n * num_groups + g] = group_var;
        }
    }
}

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

    const int total_size = N * C * spatial;
    
    // Grid-stride loop
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_size; 
         idx += blockDim.x * gridDim.x) {
        
        const int s = idx % spatial;
        const int c = (idx / spatial) % C;
        const int n = idx / (C * spatial);
        
        const int g = c / channels_per_group;
        const int stats_idx = n * num_groups + g;
        
        const scalar_t x_val = x[idx];
        const scalar_t m = mean[stats_idx];
        const scalar_t v = var[stats_idx];
        const scalar_t inv_std = rsqrt(v + eps);
        
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

    auto y = torch::empty_like(x);
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto mean = torch::empty({N, num_groups}, options);
    auto var = torch::empty({N, num_groups}, options);

    // Optimize thread block size based on problem size
    const int total_groups = N * num_groups;
    const int group_size = channels_per_group * spatial;
    const int threads_stats = min(256, nextPowerOf2(group_size));
    
    // Use multiple thread blocks for large spatial dimensions
    const int total_elements = N * C * spatial;
    const int threads_norm = 256;
    const int blocks_norm = min(1024, (total_elements + threads_norm - 1) / threads_norm);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
        compute_stats_kernel<scalar_t><<<total_groups, threads_stats, 0, stream>>>(
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
    m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA)");
}