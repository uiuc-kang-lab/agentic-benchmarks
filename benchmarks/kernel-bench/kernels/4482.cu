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
    
    // Ensure alignment for float4 loads
    const int vec_size = 4;
    const int num_vectors = group_elems / vec_size;
    const int remaining = group_elems % vec_size;
    
    scalar_t thread_sum = 0;
    scalar_t thread_sum_sq = 0;

    // Vectorized loads using __ldg for read-only data
    const float4_t* x_vec = reinterpret_cast<const float4_t*>(x + group_offset);
    #pragma unroll 2
    for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
        float4_t v = __ldg(x_vec + i);
        thread_sum += v.x + v.y + v.z + v.w;
        thread_sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Handle remaining elements with __ldg
    if (threadIdx.x < remaining) {
        const scalar_t val = __ldg(x + group_offset + num_vectors * vec_size + threadIdx.x);
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    // Warp-level reduction
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

    // Use vectorized loads where possible
    const int vec_size = 4;
    const int num_vectors = total / vec_size;
    
    // Process aligned elements using float4
    const float4_t* x_vec = reinterpret_cast<const float4_t*>(x);
    float4_t* y_vec = reinterpret_cast<float4_t*>(y);
    
    #pragma unroll 4
    for (int vid = tid; vid < num_vectors; vid += stride) {
        const int idx = vid * vec_size;
        const float4_t x_val = __ldg(x_vec + vid);
        
        // Calculate indices for the vector elements
        const int j0 = idx % spatial;
        const int temp0 = idx / spatial;
        const int c0 = temp0 % C;
        const int n0 = temp0 / C;
        const int g0 = c0 / channels_per_group;
        
        // Use __ldg for read-only accesses
        const scalar_t m = __ldg(mean + n0 * num_groups + g0);
        const scalar_t v = __ldg(var + n0 * num_groups + g0);
        const scalar_t inv_std = rsqrt(v + eps);
        const scalar_t w0 = __ldg(weight + c0);
        const scalar_t b0 = __ldg(bias + c0);
        
        float4_t result;
        result.x = (x_val.x - m) * inv_std * w0 + b0;
        
        // Handle remaining vector elements
        const int c1 = ((idx + 1) / spatial) % C;
        const int g1 = c1 / channels_per_group;
        const scalar_t w1 = __ldg(weight + c1);
        const scalar_t b1 = __ldg(bias + c1);
        result.y = (x_val.y - m) * inv_std * w1 + b1;
        
        const int c2 = ((idx + 2) / spatial) % C;
        const int g2 = c2 / channels_per_group;
        const scalar_t w2 = __ldg(weight + c2);
        const scalar_t b2 = __ldg(bias + c2);
        result.z = (x_val.z - m) * inv_std * w2 + b2;
        
        const int c3 = ((idx + 3) / spatial) % C;
        const int g3 = c3 / channels_per_group;
        const scalar_t w3 = __ldg(weight + c3);
        const scalar_t b3 = __ldg(bias + c3);
        result.w = (x_val.w - m) * inv_std * w3 + b3;
        
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
    const int threads_stats = 256;
    const dim3 blocks_stats(total_groups);

    const int threads_norm = 256;
    const int blocks_norm = (N * C * spatial + threads_norm * 4 - 1) / (threads_norm * 4);

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