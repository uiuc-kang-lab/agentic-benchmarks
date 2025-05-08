#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>

typedef float4 float4_t;

// Optimized warp reduction with no divergent branches
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized block reduction with minimal synchronization
template <typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
    static __shared__ T shared[32]; // Shared mem for 32 partial sums
    const int lid = threadIdx.x % warpSize;
    const int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val); // First reduce within warps

    if (lid == 0) shared[wid] = val; // Write reduced warp values to shared mem
    
    __syncthreads(); // Single sync point - only one needed for reduction

    // First warp reduces final results
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lid] : 0;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

// Kernel to compute per-group mean and variance
// Each block is assigned one group for one batch element
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

    // Single block reduction call handles all reductions with minimal syncs
    thread_sum = blockReduceSum(thread_sum);
    thread_sum_sq = blockReduceSum(thread_sum_sq);

    if (threadIdx.x == 0) {
        const scalar_t group_mean = thread_sum / group_elems;
        const scalar_t group_var = thread_sum_sq / group_elems - group_mean * group_mean;
        const int out_index = n * num_groups + g;
        mean[out_index] = group_mean;
        var[out_index] = group_var;
    }
}

// Kernel to apply the group normalization
// Each thread processes one element from the input
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
                const int stats_index = n * num_groups + g;

                const scalar_t m = __ldg(mean + stats_index);
                const scalar_t v = __ldg(var + stats_index);
                const scalar_t inv_std = rsqrt(v + eps);
                const scalar_t w = __ldg(weight + c);
                const scalar_t b = __ldg(bias + c);

                (&result.x)[i] = ((&x_val.x)[i] - m) * inv_std * w + b;
            }
        }
        
        *reinterpret_cast<float4_t*>(y + base_idx) = result;
    }
}

// Host function to launch the optimized kernels with CUDA streams
// This function uses multiple streams to overlap kernel execution with memory transfers
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
    const int threads_stats = 512; // Increased thread count for better occupancy
    const dim3 blocks_stats(total_groups);

    const int threads_norm = 256;
    const int blocks_norm = (N * C * spatial + threads_norm * 4 - 1) / (threads_norm * 4);

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
        // Launch the kernel to compute means and variances on stream1
        compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, 0, stream1>>>(
            x.data_ptr<scalar_t>(),
            N, C, spatial,
            channels_per_group,
            num_groups,
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>());

        // Launch the kernel to perform group normalization on stream2
        group_norm_forward_kernel<scalar_t><<<blocks_norm, threads_norm, 0, stream2>>>(
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

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with pipelined streams");
}
