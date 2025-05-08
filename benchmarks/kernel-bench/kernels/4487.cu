#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

#define BLOCK_SIZE_STATS 128  // Reduced block size for stats computation
#define BLOCK_SIZE_NORM 512   // Increased block size for normalization

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
    
    // Process multiple elements per thread to compensate for smaller block size
    const int elements_per_thread = 4;
    const int vec_size = 4;
    const int num_vectors = group_elems / vec_size;
    const int vectors_per_thread = (num_vectors + blockDim.x - 1) / blockDim.x;
    
    scalar_t thread_sum[4] = {0, 0, 0, 0};
    scalar_t thread_sum_sq[4] = {0, 0, 0, 0};

    const float4_t* x_vec = reinterpret_cast<const float4_t*>(x + group_offset);
    
    #pragma unroll
    for (int i = 0; i < vectors_per_thread; i++) {
        const int vid = threadIdx.x + i * blockDim.x;
        if (vid < num_vectors) {
            float4_t v = __ldg(x_vec + vid);
            thread_sum[0] += v.x;
            thread_sum[1] += v.y;
            thread_sum[2] += v.z;
            thread_sum[3] += v.w;
            thread_sum_sq[0] += v.x * v.x;
            thread_sum_sq[1] += v.y * v.y;
            thread_sum_sq[2] += v.z * v.z;
            thread_sum_sq[3] += v.w * v.w;
        }
    }

    // Combine partial sums
    scalar_t final_sum = 0;
    scalar_t final_sum_sq = 0;
    
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        final_sum += thread_sum[i];
        final_sum_sq += thread_sum_sq[i];
    }

    // Handle remaining elements
    const int remaining = group_elems % vec_size;
    if (threadIdx.x < remaining) {
        const scalar_t val = __ldg(x + group_offset + num_vectors * vec_size + threadIdx.x);
        final_sum += val;
        final_sum_sq += val * val;
    }

    // Warp reduction
    final_sum = warpReduceSum(final_sum);
    final_sum_sq = warpReduceSum(final_sum_sq);

    __shared__ scalar_t s_partial_sums[4];    // Reduced shared memory size
    __shared__ scalar_t s_partial_squares[4];

    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;

    if (lane_id == 0) {
        s_partial_sums[warp_id] = final_sum;
        s_partial_squares[warp_id] = final_sum_sq;
    }
    __syncthreads();

    // Final reduction with first warp
    if (threadIdx.x < 4) {  // Only need first 4 threads for final reduction
        final_sum = s_partial_sums[threadIdx.x];
        final_sum_sq = s_partial_squares[threadIdx.x];
    }
    
    final_sum = warpReduceSum(final_sum);
    final_sum_sq = warpReduceSum(final_sum_sq);

    if (threadIdx.x == 0) {
        const scalar_t group_mean = final_sum / group_elems;
        const scalar_t group_var = final_sum_sq / group_elems - group_mean * group_mean;
        const int out_index = n * num_groups + g;
        mean[out_index] = group_mean;
        var[out_index] = group_var;
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

    // Process 4 elements per thread using vectorized loads
    const int vec_size = 4;
    const int num_vectors = total / vec_size;
    
    const float4_t* x_vec = reinterpret_cast<const float4_t*>(x);
    float4_t* y_vec = reinterpret_cast<float4_t*>(y);
    
    #pragma unroll 4
    for (int vid = tid; vid < num_vectors; vid += stride) {
        const float4_t x_val = __ldg(x_vec + vid);
        const int base_idx = vid * vec_size;

        float4_t result;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = base_idx + i;
            const int j = idx % spatial;
            const int temp = idx / spatial;
            const int c = temp % C;
            const int n = temp / C;
            const int g = c / channels_per_group;
            const int stats_index = n * num_groups + g;

            const scalar_t m = __ldg(mean + stats_index);
            const scalar_t v = __ldg(var + stats_index);
            const scalar_t inv_std = rsqrt(v + eps);
            const scalar_t w = __ldg(weight + c);
            const scalar_t b = __ldg(bias + c);

            reinterpret_cast<scalar_t*>(&result)[i] = 
                (reinterpret_cast<const scalar_t*>(&x_val)[i] - m) * inv_std * w + b;
        }
        y_vec[vid] = result;
    }

    // Handle remaining elements
    for (int idx = tid + num_vectors * vec_size; idx < total; idx += stride) {
        const int j = idx % spatial;
        const int temp = idx / spatial;
        const int c = temp % C;
        const int n = temp / C;
        const int g = c / channels_per_group;
        const int stats_index = n * num_groups + g;
        
        const scalar_t m = __ldg(mean + stats_index);
        const scalar_t v = __ldg(var + stats_index);
        const scalar_t inv_std = rsqrt(v + eps);
        const scalar_t x_val = __ldg(x + idx);
        const scalar_t w = __ldg(weight + c);
        const scalar_t b = __ldg(bias + c);
        
        y[idx] = (x_val - m) * inv_std * w + b;
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
    const dim3 blocks_stats(total_groups);
    const int threads_stats = BLOCK_SIZE_STATS;

    const int total_elements = N * C * spatial;
    const int threads_norm = BLOCK_SIZE_NORM;
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