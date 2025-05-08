#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <algorithm>

// Use vectorized loads for coalesced memory access
typedef float4 float4_t;

// Optimized warp reduction
template <typename T>
__device__ __forceinline__ T warpReduceSum(T val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized block reduction with minimal synchronization
template <typename T>
__device__ __forceinline__ T blockReduceSum(T val) {
    __shared__ T shared[32];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    val = warpReduceSum(val);
    if (lane == 0) {
        shared[warpId] = val;
    }
    __syncthreads();
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
    if (warpId == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Kernel to compute per-group mean and variance for a tile (mini-batch slice).
// Each block processes one (n, g) pair in the tile.
template <typename scalar_t>
__global__ void compute_stats_kernel(
    const scalar_t* __restrict__ x,
    const int N,                 // number of samples in the tile
    const int C,
    const int spatial,           // product of dimensions from index 2 onward
    const int channels_per_group,
    const int num_groups,
    scalar_t* __restrict__ mean, // output: (N, num_groups)
    scalar_t* __restrict__ var)  // output: (N, num_groups)
{
    // Each block corresponds to one (n, g) pair in the tile
    int idx = blockIdx.x;
    int n = idx / num_groups;
    int g = idx % num_groups;

    int group_offset = n * C * spatial + g * channels_per_group * spatial;
    int group_elems = channels_per_group * spatial;

    // Use vectorized loads
    int vec_size = 4;
    int num_vectors = group_elems / vec_size;
    int remaining = group_elems % vec_size;

    scalar_t thread_sum = 0;
    scalar_t thread_sum_sq = 0;

    const float4_t* x_vec = reinterpret_cast<const float4_t*>(x + group_offset);
    for (int i = threadIdx.x; i < num_vectors; i += blockDim.x) {
        float4_t v = __ldg(x_vec + i);
        thread_sum += v.x + v.y + v.z + v.w;
        thread_sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    if (threadIdx.x < remaining) {
        int index_rem = num_vectors * vec_size + threadIdx.x;
        scalar_t val = __ldg(x + group_offset + index_rem);
        thread_sum += val;
        thread_sum_sq += val * val;
    }

    thread_sum = blockReduceSum(thread_sum);
    thread_sum_sq = blockReduceSum(thread_sum_sq);

    if (threadIdx.x == 0) {
        scalar_t m = thread_sum / group_elems;
        scalar_t v_val = thread_sum_sq / group_elems - m * m;
        int out_index = n * num_groups + g;
        mean[out_index] = m;
        var[out_index] = v_val;
    }
}

// Kernel to apply group normalization for a tile
// Each thread processes multiple elements via grid-stride loop
template <typename scalar_t>
__global__ void group_norm_forward_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ mean,
    const scalar_t* __restrict__ var,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int N,                // number of samples in the tile
    const int C,
    const int spatial,          // product of dimensions from index 2 onward
    const int channels_per_group,
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * spatial;
    int stride = blockDim.x * gridDim.x;

    // Process 4 elements per iteration
    for (int base_idx = tid * 4; base_idx < total; base_idx += stride * 4) {
        float4_t x_val = __ldg(reinterpret_cast<const float4_t*>(x + base_idx));
        float4_t result;

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            int idx = base_idx + i;
            if (idx < total) {
                int j = idx % spatial;
                int temp = idx / spatial;
                int c = temp % C;
                int n = temp / C;
                int g = c / channels_per_group;
                int stats_index = n * num_groups + g;
                scalar_t m = __ldg(mean + stats_index);
                scalar_t var_val = __ldg(var + stats_index);
                scalar_t inv_std = rsqrt(var_val + eps);
                scalar_t w = __ldg(weight + c);
                scalar_t b = __ldg(bias + c);
                ((&result.x))[i] = (((&x_val.x))[i] - m) * inv_std * w + b;
            }
        }
        *reinterpret_cast<float4_t*>(y + base_idx) = result;
    }
}

// Host function implementing pipelined tiling to overlap computation and memory operations.
// The input batch is processed in tiles, with compute_stats_kernel launched in one stream
// and group_norm_forward_kernel in another stream, using events for synchronization.

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

    // Input x is expected to have shape (N, C, ...).
    const int N = x.size(0);
    const int C = x.size(1);
    int spatial = 1;
    for (int i = 2; i < x.dim(); i++) {
        spatial *= x.size(i);
    }
    int channels_per_group = C / num_groups;

    auto y = torch::empty_like(x);
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto mean = torch::empty({N, num_groups}, options);
    auto var  = torch::empty({N, num_groups}, options);

    // Define tile size along the batch dimension for pipelining
    int tile_size = 8;

    cudaStream_t stream_compute, stream_forward;
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_forward);
    cudaEvent_t compute_done;
    cudaEventCreate(&compute_done);

    for (int b = 0; b < N; b += tile_size) {
        int curr_tile = std::min(tile_size, N - b);
        int x_offset = b * C * spatial;         // Offset for input and output x/y
        int y_offset = x_offset;
        int stats_offset = b * num_groups;        // Offset for mean and var

        // Launch compute_stats_kernel for the current tile on stream_compute
        int total_groups_tile = curr_tile * num_groups;
        int threads_stats = 512;
        dim3 blocks_stats(total_groups_tile);

        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "compute_stats_tile_cuda", ([&] {
            compute_stats_kernel<scalar_t><<<blocks_stats, threads_stats, 0, stream_compute>>>(
                x.data_ptr<scalar_t>() + x_offset,
                curr_tile, C, spatial, channels_per_group, num_groups,
                mean.data_ptr<scalar_t>() + stats_offset,
                var.data_ptr<scalar_t>() + stats_offset);
        }));

        // Record an event on stream_compute after stats computation for this tile
        cudaEventRecord(compute_done, stream_compute);
        // Make stream_forward wait until the compute_stats_kernel finishes for this tile
        cudaStreamWaitEvent(stream_forward, compute_done, 0);

        // Launch group_norm_forward_kernel for this tile on stream_forward
        int total_tile_elements = curr_tile * C * spatial;
        int threads_norm = 256;
        int blocks_norm = (total_tile_elements + threads_norm * 4 - 1) / (threads_norm * 4);
        AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_tile_cuda", ([&] {
            group_norm_forward_kernel<scalar_t><<<blocks_norm, threads_norm, 0, stream_forward>>>(
                x.data_ptr<scalar_t>() + x_offset,
                mean.data_ptr<scalar_t>() + stats_offset,
                var.data_ptr<scalar_t>() + stats_offset,
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                curr_tile, C, spatial, channels_per_group, num_groups,
                static_cast<scalar_t>(eps),
                y.data_ptr<scalar_t>() + y_offset);
        }));
    }

    cudaStreamSynchronize(stream_compute);
    cudaStreamSynchronize(stream_forward);
    cudaEventDestroy(compute_done);
    cudaStreamDestroy(stream_compute);
    cudaStreamDestroy(stream_forward);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &group_norm_forward, "Group Normalization forward (CUDA) with pipelined tiling for overlapping computation and memory transfers");
}
