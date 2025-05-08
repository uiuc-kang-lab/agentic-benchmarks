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
    
    // Shared memory for two-level reduction
    extern __shared__ char shared[];
    scalar_t* s_data = reinterpret_cast<scalar_t*>(shared);
    scalar_t* s_data_sq = s_data + blockDim.x;
    
    const int vec_size = 4;
    const int num_vectors = group_elems / vec_size;
    const int remaining = group_elems % vec_size;
    
    scalar_t thread_sum = 0;
    scalar_t thread_sum_sq = 0;

    // Vectorized loads using __ldg
    const float4_t* x_vec = reinterpret_cast<const float4_t*>(x + group_offset);
    #pragma unroll 4
    for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
        float4_t v = __ldg(x_vec + i);
        thread_sum += v.x + v.y + v.z + v.w;
        thread_sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    if (threadIdx.x < remaining) {
        const scalar_t val = __ldg(x + group_offset + num_vectors * vec_size + threadIdx.x);
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    // Store partial sums in shared memory
    s_data[threadIdx.x] = thread_sum;
    s_data_sq[threadIdx.x] = thread_sum_sq;
    __syncthreads();

    // Two-level reduction: first within warps, then across warps
    if (threadIdx.x < warpSize) {
        scalar_t warp_sum = 0;
        scalar_t warp_sum_sq = 0;
        
        #pragma unroll
        for (int i = threadIdx.x; i < blockDim.x; i += warpSize) {
            warp_sum += s_data[i];
            warp_sum_sq += s_data_sq[i];
        }
        
        // Warp-level reduction using shuffle
        warp_sum = warpReduceSum(warp_sum);
        warp_sum_sq = warpReduceSum(warp_sum_sq);

        if (threadIdx.x == 0) {
            const scalar_t group_mean = warp_sum / group_elems;
            const scalar_t group_var = warp_sum_sq / group_elems - group_mean * group_mean;
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

    // Shared memory for caching frequently accessed data
    extern __shared__ char shared[];
    scalar_t* s_mean = reinterpret_cast<scalar_t*>(shared);
    scalar_t* s_var = s_mean + num_groups;
    
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    
    // Cache group statistics in shared memory
    if (threadIdx.x < num_groups) {
        const int group_idx = blockIdx.x * num_groups + threadIdx.x;
        if (group_idx < N * num_groups) {
            s_mean[threadIdx.x] = __ldg(mean + group_idx);
            s_var[threadIdx.x] = __ldg(var + group_idx);
        }
    }
    __syncthreads();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total = N * C * spatial;

    // Process 4 elements per thread per iteration
    #pragma unroll 4
    for (int base_idx = tid * 4; base_idx < total; base_idx += stride * 4) {
        float4_t x_val = __ldg(reinterpret_cast<const float4_t*>(x + base_idx));
        float4_t result;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int idx = base_idx + i;
            if (idx < total) {
                const int j = idx % spatial;
                const int temp = idx / spatial;
                const int c = temp % C;
                const int n = temp / C;
                const int g = c / channels_per_group;
                const int local_g = g % num_groups;

                const scalar_t m = s_mean[local_g];
                const scalar_t v = s_var[local_g];
                const scalar_t inv_std = rsqrt(v + eps);
                const scalar_t w = __ldg(weight + c);
                const scalar_t b = __ldg(bias + c);

                (&result.x)[i] = ((&x_val.x)[i] - m) * inv_std * w + b;
            }
        }
        
        *reinterpret_cast<float4_t*>(y + base_idx) = result;
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
    const dim3 blocks_stats(total_groups);
    const size_t shared_mem_stats = threads_stats * 2 * sizeof(float);

    const int threads_norm = 256;
    const dim3 blocks_norm((N * C * spatial + threads_norm * 4 - 1) / (threads_norm * 4));
    const size_t shared_mem_norm = num_groups * 2 * sizeof(float);

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
        compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, shared_mem_stats, stream>>>(
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