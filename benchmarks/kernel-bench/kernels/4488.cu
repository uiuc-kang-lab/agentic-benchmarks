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
    
    // Shared memory for block-level reduction
    extern __shared__ char shared[];
    scalar_t* s_data = reinterpret_cast<scalar_t*>(shared);
    scalar_t* s_data_sq = s_data + blockDim.x;
    
    scalar_t thread_sum = 0;
    scalar_t thread_sum_sq = 0;

    // Process 4 elements per thread per iteration
    const int vec_size = 4;
    const int num_vectors = group_elems / vec_size;
    const float4_t* x_vec = reinterpret_cast<const float4_t*>(x + group_offset);
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
        float4_t v = __ldg(x_vec + i);
        thread_sum += v.x + v.y + v.z + v.w;
        thread_sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Handle remaining elements
    const int base = num_vectors * vec_size;
    #pragma unroll
    for (int i = threadIdx.x; i < group_elems - base; i += blockDim.x) {
        const scalar_t val = __ldg(x + group_offset + base + i);
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    // Store in shared memory
    s_data[threadIdx.x] = thread_sum;
    s_data_sq[threadIdx.x] = thread_sum_sq;
    __syncthreads();

    // Block-level reduction
    #pragma unroll
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
            s_data_sq[threadIdx.x] += s_data_sq[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Final warp reduction
    if (threadIdx.x < 32) {
        thread_sum = s_data[threadIdx.x];
        thread_sum_sq = s_data_sq[threadIdx.x];
        
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
            thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
        }

        if (threadIdx.x == 0) {
            const scalar_t group_mean = thread_sum / group_elems;
            const scalar_t group_var = thread_sum_sq / group_elems - group_mean * group_mean;
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

    // Process multiple elements per thread using float4
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = N * C * spatial;
    const int vec_size = 4;
    const int total_vectors = total / vec_size;

    // Shared memory cache for frequently accessed data
    extern __shared__ char shared[];
    scalar_t* s_weight = reinterpret_cast<scalar_t*>(shared);
    scalar_t* s_bias = s_weight + channels_per_group;

    // Cache weight and bias for current group
    const int local_c = (blockIdx.x * blockDim.x) / spatial % C;
    const int local_g = local_c / channels_per_group;
    if (threadIdx.x < channels_per_group) {
        s_weight[threadIdx.x] = __ldg(weight + local_g * channels_per_group + threadIdx.x);
        s_bias[threadIdx.x] = __ldg(bias + local_g * channels_per_group + threadIdx.x);
    }
    __syncthreads();

    // Process aligned data with float4
    const float4_t* x_vec = reinterpret_cast<const float4_t*>(x);
    float4_t* y_vec = reinterpret_cast<float4_t*>(y);

    #pragma unroll 4
    for (int vid = tid; vid < total_vectors; vid += stride) {
        const int idx = vid * vec_size;
        const float4_t x_val = __ldg(x_vec + vid);
        
        // Calculate indices
        const int j = idx % spatial;
        const int temp = idx / spatial;
        const int c = temp % C;
        const int n = temp / C;
        const int g = c / channels_per_group;
        
        // Load stats using __ldg
        const int stats_idx = n * num_groups + g;
        const scalar_t m = __ldg(mean + stats_idx);
        const scalar_t v = __ldg(var + stats_idx);
        const scalar_t inv_std = rsqrt(v + eps);

        // Process all 4 elements
        float4_t result;
        const int local_offset = c % channels_per_group;
        result.x = (x_val.x - m) * inv_std * s_weight[local_offset] + s_bias[local_offset];
        result.y = (x_val.y - m) * inv_std * s_weight[local_offset] + s_bias[local_offset];
        result.z = (x_val.z - m) * inv_std * s_weight[local_offset] + s_bias[local_offset];
        result.w = (x_val.w - m) * inv_std * s_weight[local_offset] + s_bias[local_offset];
        
        y_vec[vid] = result;
    }

    // Handle remaining elements
    #pragma unroll
    for (int idx = tid * vec_size + total_vectors * vec_size; idx < total; idx++) {
        const int j = idx % spatial;
        const int temp = idx / spatial;
        const int c = temp % C;
        const int n = temp / C;
        const int g = c / channels_per_group;
        
        const scalar_t m = __ldg(mean + n * num_groups + g);
        const scalar_t v = __ldg(var + n * num_groups + g);
        const scalar_t inv_std = rsqrt(v + eps);
        const scalar_t x_val = __ldg(x + idx);
        
        const int local_offset = c % channels_per_group;
        y[idx] = (x_val - m) * inv_std * s_weight[local_offset] + s_bias[local_offset];
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
    const int threads_stats = 512;
    const size_t shared_mem_stats = threads_stats * 2 * sizeof(float);
    
    const int threads_norm = 256;
    const size_t shared_mem_norm = channels_per_group * 2 * sizeof(float);
    const int blocks_norm = (N * C * spatial + threads_norm * 4 - 1) / (threads_norm * 4);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
        compute_stats_kernel<scalar_t><<<total_groups, threads_stats, shared_mem_stats, stream>>>(
            x.data_ptr<scalar_t>(),
            N, C, spatial,
            channels_per_group,
            num_groups,
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>());

        group_norm_forward_kernel<scalar_t><<<blocks_norm, threads_norm, shared_mem_norm, stream>>>(
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