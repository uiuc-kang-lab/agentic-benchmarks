#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

typedef float4 float4_t;

template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
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

    const int idx = blockIdx.x;
    const int n = idx / num_groups;
    const int g = idx % num_groups;
    
    const int group_offset = n * C * spatial + g * channels_per_group * spatial;
    const int group_elems = channels_per_group * spatial;
    
    // Process 4 elements per vector load
    const int vec_size = 4;
    const int num_vectors = group_elems / vec_size;
    const int remaining = group_elems % vec_size;
    
    scalar_t thread_sum[4] = {0, 0, 0, 0};
    scalar_t thread_sum_sq[4] = {0, 0, 0, 0};

    // Manual unroll of vector loads
    const float4_t* x_vec = reinterpret_cast<const float4_t*>(x + group_offset);
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
        float4_t v = x_vec[i];
        thread_sum[0] += v.x;
        thread_sum[1] += v.y;
        thread_sum[2] += v.z;
        thread_sum[3] += v.w;
        thread_sum_sq[0] += v.x * v.x;
        thread_sum_sq[1] += v.y * v.y;
        thread_sum_sq[2] += v.z * v.z;
        thread_sum_sq[3] += v.w * v.w;
    }

    // Combine partial sums
    scalar_t final_sum = 0;
    scalar_t final_sum_sq = 0;
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        final_sum += thread_sum[i];
        final_sum_sq += thread_sum_sq[i];
    }

    // Handle remaining elements
    #pragma unroll
    for (int i = 0; i < remaining; ++i) {
        const scalar_t val = x[group_offset + num_vectors * vec_size + i];
        final_sum += val;
        final_sum_sq += val * val;
    }

    final_sum = warpReduceSum(final_sum);
    final_sum_sq = warpReduceSum(final_sum_sq);

    __shared__ scalar_t s_partial_sums[32];
    __shared__ scalar_t s_partial_squares[32];

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    if (lane_id == 0) {
        s_partial_sums[warp_id] = final_sum;
        s_partial_squares[warp_id] = final_sum_sq;
    }
    __syncthreads();

    if (warp_id == 0 && lane_id < (blockDim.x + warpSize - 1) / warpSize) {
        final_sum = s_partial_sums[lane_id];
        final_sum_sq = s_partial_squares[lane_id];
        
        final_sum = warpReduceSum(final_sum);
        final_sum_sq = warpReduceSum(final_sum_sq);

        if (lane_id == 0) {
            const scalar_t group_mean = final_sum / group_elems;
            const scalar_t group_var = final_sum_sq / group_elems - group_mean * group_mean;
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

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = N * C * spatial;

    // Process 4 elements per thread per iteration
    #pragma unroll 4
    for (int base_idx = tid * 4; base_idx < total; base_idx += stride * 4) {
        scalar_t x_vals[4];
        scalar_t y_vals[4];
        int indices[4];
        int channels[4];
        int samples[4];
        int groups[4];
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = base_idx + i;
            if (idx < total) {
                indices[i] = idx;
                const int j = idx % spatial;
                const int temp = idx / spatial;
                channels[i] = temp % C;
                samples[i] = temp / C;
                groups[i] = channels[i] / channels_per_group;
                x_vals[i] = x[idx];
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = base_idx + i;
            if (idx < total) {
                const int stats_index = samples[i] * num_groups + groups[i];
                const scalar_t m = mean[stats_index];
                const scalar_t v = var[stats_index];
                const scalar_t inv_std = rsqrt(v + eps);
                y_vals[i] = (x_vals[i] - m) * inv_std * weight[channels[i]] + bias[channels[i]];
            }
        }

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = base_idx + i;
            if (idx < total) {
                y[idx] = y_vals[i];
            }
        }
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

    const int total_groups = N * num_groups;
    const int threads_stats = 256;
    const dim3 blocks_stats(total_groups);

    const int total_elements = N * C * spatial;
    const int threads_norm = 256;
    const dim3 blocks_norm((total_elements + threads_norm * 4 - 1) / (threads_norm * 4));

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
        compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            N, C, spatial,
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