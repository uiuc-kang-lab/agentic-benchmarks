#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,
    const int C,
    const int spatial,
    const int channels_per_group,
    const int num_groups,
    const int elements_per_block,
    scalar_t* __restrict__ partial_sums,
    scalar_t* __restrict__ partial_squares) {

    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int n = blockIdx.y;
    const int g = blockIdx.z;
    
    const int group_size = channels_per_group * spatial;
    const int start_idx = bid * elements_per_block;
    const int group_offset = n * C * spatial + g * channels_per_group * spatial;
    
    __shared__ scalar_t s_sum[256];
    __shared__ scalar_t s_sum_sq[256];
    
    s_sum[tid] = 0;
    s_sum_sq[tid] = 0;

    if (((size_t)(x + group_offset + start_idx) & 15) == 0 && spatial % 4 == 0) {
        const float4* x4 = reinterpret_cast<const float4*>(x + group_offset + start_idx);
        for (int i = tid; i < elements_per_block/4 && (start_idx + i*4) < group_size; i += blockDim.x) {
            float4 val4 = __ldg(x4 + i);
            s_sum[tid] += val4.x + val4.y + val4.z + val4.w;
            s_sum_sq[tid] += val4.x * val4.x + val4.y * val4.y + val4.z * val4.z + val4.w * val4.w;
        }
    } else {
        for (int i = tid; i < elements_per_block && (start_idx + i) < group_size; i += blockDim.x) {
            const scalar_t val = __ldg(x + group_offset + start_idx + i);
            s_sum[tid] += val;
            s_sum_sq[tid] += val * val;
        }
    }
    __syncthreads();
    
        __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        const int partial_idx = n * (num_groups * gridDim.x) + g * gridDim.x + bid;
        partial_sums[partial_idx] = s_sum[0];
        partial_squares[partial_idx] = s_sum_sq[0];
    }
}

template <typename scalar_t>
__global__ void finalize_stats_kernel(
    const scalar_t* __restrict__ partial_sums,
    const scalar_t* __restrict__ partial_squares,
    const int num_blocks,
    const int group_size,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var) {
    
    const int n = blockIdx.y;
    const int g = threadIdx.x;
    
    if (g >= gridDim.x) return;
    
    scalar_t sum = 0;
    scalar_t sum_sq = 0;
    
    #pragma unroll 4
    for (int i = 0; i < num_blocks; i++) {
        const int idx = n * (gridDim.x * num_blocks) + g * num_blocks + i;
        sum += __ldg(&partial_sums[idx]);
        sum_sq += __ldg(&partial_squares[idx]);
    }
    
    const scalar_t group_mean = sum / group_size;
    const scalar_t group_var = sum_sq / group_size - group_mean * group_mean;
    const int out_idx = n * gridDim.x + g;
    mean[out_idx] = group_mean;
    var[out_idx] = group_var;
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
    
    if (((size_t)x & 15) == 0 && ((size_t)y & 15) == 0 && spatial % 4 == 0) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        float4* y4 = reinterpret_cast<float4*>(y);
        
        for (int idx = tid; idx < total/4; idx += stride) {
            const int base_idx = idx * 4;
            const int j = base_idx % spatial;
            const int temp = base_idx / spatial;
            const int c = temp % C;
            const int n = temp / C;
            
            const int g = c / channels_per_group;
            const int stats_idx = n * num_groups + g;
            
            const scalar_t m = __ldg(&mean[stats_idx]);
            const scalar_t v = __ldg(&var[stats_idx]);
            const scalar_t inv_std = rsqrt(v + eps);
            
            float4 val4 = __ldg(&x4[idx]);
            float4 result;
            result.x = (val4.x - m) * inv_std * __ldg(&weight[c]) + __ldg(&bias[c]);
            result.y = (val4.y - m) * inv_std * __ldg(&weight[c+1]) + __ldg(&bias[c+1]);
            result.z = (val4.z - m) * inv_std * __ldg(&weight[c+2]) + __ldg(&bias[c+2]);
            result.w = (val4.w - m) * inv_std * __ldg(&weight[c+3]) + __ldg(&bias[c+3]);
            y4[idx] = result;
        }
    } else {
        for (int idx = tid; idx < total; idx += stride) {
            const int j = idx % spatial;
            const int temp = idx / spatial;
            const int c = temp % C;
            const int n = temp / C;
            
            const int g = c / channels_per_group;
            const int stats_idx = n * num_groups + g;
            
            const scalar_t m = __ldg(&mean[stats_idx]);
            const scalar_t v = __ldg(&var[stats_idx]);
            const scalar_t inv_std = rsqrt(v + eps);
            const scalar_t x_val = __ldg(&x[idx]);
            y[idx] = (x_val - m) * inv_std * __ldg(&weight[c]) + __ldg(&bias[c]);
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
    const int group_size = channels_per_group * spatial;
    const int block_size = 256;
    const int elements_per_block = 1024;
    const int num_blocks = (group_size + elements_per_block - 1) / elements_per_block;
    
    auto y = torch::empty_like(x);
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto mean = torch::empty({N, num_groups}, options);
    auto var = torch::empty({N, num_groups}, options);
    auto partial_sums = torch::empty({N, num_groups, num_blocks}, options);
    auto partial_squares = torch::empty({N, num_groups, num_blocks}, options);
    
    dim3 stats_blocks(num_blocks, N, num_groups);
    dim3 norm_blocks((N * C * spatial + block_size - 1) / block_size);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
        compute_stats_kernel<scalar_t><<<stats_blocks, block_size, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            N, C, spatial,
            channels_per_group,
            num_groups,
            elements_per_block,
            partial_sums.data_ptr<scalar_t>(),
            partial_squares.data_ptr<scalar_t>());
            
        finalize_stats_kernel<scalar_t><<<dim3(1, N), num_groups, 0, stream>>>(
            partial_sums.data_ptr<scalar_t>(),
            partial_squares.data_ptr<scalar_t>(),
            num_blocks,
            group_size,
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>());
            
        group_norm_forward_kernel<scalar_t><<<norm_blocks, block_size, 0, stream>>>(
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