#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define BLOCK_SIZE_STATS 256
#define BLOCK_SIZE_NORM 256
#define ITEMS_PER_THREAD 4

template <typename scalar_t, int BLOCK_SIZE>
__global__ void compute_stats_kernel_atomic(
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

    // Use registers for local accumulation
    scalar_t local_sum = 0;
    scalar_t local_sum_sq = 0;

    // Each thread processes multiple elements
    #pragma unroll
    for (int i = threadIdx.x; i < group_elems; i += BLOCK_SIZE * ITEMS_PER_THREAD) {
        scalar_t thread_vals[ITEMS_PER_THREAD];
        #pragma unroll
        for (int j = 0; j < ITEMS_PER_THREAD; j++) {
            const int curr_idx = i + j;
            if (curr_idx < group_elems) {
                const int c = curr_idx / spatial;
                const int s = curr_idx % spatial;
                thread_vals[j] = x[group_offset + c * spatial + s];
                local_sum += thread_vals[j];
                local_sum_sq += thread_vals[j] * thread_vals[j];
            }
        }
    }

    // Shared memory for block reduction
    extern __shared__ char smem[];
    scalar_t* s_sum = reinterpret_cast<scalar_t*>(smem);
    scalar_t* s_sum_sq = s_sum + BLOCK_SIZE;

    s_sum[threadIdx.x] = local_sum;
    s_sum_sq[threadIdx.x] = local_sum_sq;
    __syncthreads();

    // Efficient block reduction
    #pragma unroll
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_sum[threadIdx.x] += s_sum[threadIdx.x + stride];
            s_sum_sq[threadIdx.x] += s_sum_sq[threadIdx.x + stride];
        }
        if (stride > 32) __syncthreads();
    }

    // Only first thread writes results
    if (threadIdx.x == 0) {
        scalar_t group_mean = s_sum[0] / group_elems;
        scalar_t group_var = s_sum_sq[0] / group_elems - group_mean * group_mean;
        int out_index = n * num_groups + g;
        mean[out_index] = group_mean;
        var[out_index] = group_var;
    }
}

template <typename scalar_t, int BLOCK_SIZE>
__global__ void group_norm_forward_kernel_atomic(
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

    // Process multiple elements per thread
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = N * C * spatial;

    #pragma unroll
    for (int idx = tid; idx < total; idx += stride) {
        const int j = idx % spatial;
        const int temp = idx / spatial;
        const int c = temp % C;
        const int n = temp / C;
        
        const int g = c / channels_per_group;
        const int stats_index = n * num_groups + g;
        
        const scalar_t m = mean[stats_index];
        const scalar_t v = var[stats_index];
        const scalar_t inv_std = rsqrt(v + eps);
        const scalar_t x_val = x[idx];
        y[idx] = (x_val - m) * inv_std * weight[c] + bias[c];
    }
}

torch::Tensor group_norm_forward_atomic(
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

    const int total_groups = N * num_groups;
    const int threads_stats = BLOCK_SIZE_STATS;
    const int total_elements = N * C * spatial;
    const int blocks_norm = (total_elements + BLOCK_SIZE_NORM * ITEMS_PER_THREAD - 1) / (BLOCK_SIZE_NORM * ITEMS_PER_THREAD);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_atomic_cuda", ([&] {
        size_t shared_mem_size = BLOCK_SIZE_STATS * 2 * sizeof(scalar_t);

        compute_stats_kernel_atomic<scalar_t, BLOCK_SIZE_STATS><<<
            total_groups, threads_stats, shared_mem_size, stream>>>(
            x.data_ptr<scalar_t>(),
            N, C, spatial,
            channels_per_group,
            num_groups,
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>());

        group_norm_forward_kernel_atomic<scalar_t, BLOCK_SIZE_NORM><<<
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
    m.def("forward", &group_norm_forward_atomic, "Atomic optimized Group Normalization forward (CUDA)");
}