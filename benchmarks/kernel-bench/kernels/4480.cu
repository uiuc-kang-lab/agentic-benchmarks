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
    const int H,
    const int W,
    const int channels_per_group,
    const int num_groups,
    scalar_t* __restrict__ mean,
    scalar_t* __restrict__ var) {

    // Use 2D thread blocks for spatial dimensions
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int n = blockIdx.z;  // batch dimension
    const int g = blockIdx.y;  // group dimension
    
    const int TILE_DIM = 16;  // Thread block tile dimensions
    extern __shared__ char smem[];
    scalar_t* s_data = reinterpret_cast<scalar_t*>(smem);
    
    const int spatial_size = H * W;
    const int group_start = g * channels_per_group;
    const int group_end = (g + 1) * channels_per_group;
    
    scalar_t thread_sum = 0;
    scalar_t thread_sum_sq = 0;

    // Grid-stride loop over channels and spatial dimensions
    for (int c = group_start; c < group_end; c++) {
        for (int h = ty; h < H; h += blockDim.y) {
            for (int w = tx; w < W; w += blockDim.x) {
                const int idx = ((n * C + c) * H + h) * W + w;
                const scalar_t val = x[idx];
                thread_sum += val;
                thread_sum_sq += val * val;
            }
        }
    }

    // Warp reduction
    thread_sum = warpReduceSum(thread_sum);
    thread_sum_sq = warpReduceSum(thread_sum_sq);

    // Block reduction using shared memory
    const int tid = ty * blockDim.x + tx;
    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    
    if (lane_id == 0) {
        s_data[warp_id] = thread_sum;
        s_data[warp_id + blockDim.x * blockDim.y / warpSize] = thread_sum_sq;
    }
    __syncthreads();

    // Final reduction by first warp
    if (tid < (blockDim.x * blockDim.y / warpSize)) {
        thread_sum = s_data[tid];
        thread_sum_sq = s_data[tid + blockDim.x * blockDim.y / warpSize];
        
        thread_sum = warpReduceSum(thread_sum);
        thread_sum_sq = warpReduceSum(thread_sum_sq);
        
        if (tid == 0) {
            const int group_size = channels_per_group * spatial_size;
            const scalar_t group_mean = thread_sum / group_size;
            const scalar_t group_var = thread_sum_sq / group_size - group_mean * group_mean;
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
    const int H,
    const int W,
    const int channels_per_group,
    const int num_groups,
    const scalar_t eps,
    scalar_t* __restrict__ y) {

    // 3D grid for direct mapping to data dimensions
    const int w = blockIdx.x * blockDim.x + threadIdx.x;
    const int h = blockIdx.y * blockDim.y + threadIdx.y;
    const int nc = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (w >= W || h >= H || nc >= N * C) return;
    
    const int n = nc / C;
    const int c = nc % C;
    const int g = c / channels_per_group;
    
    const int stats_index = n * num_groups + g;
    const scalar_t m = mean[stats_index];
    const scalar_t v = var[stats_index];
    const scalar_t inv_std = rsqrt(v + eps);
    
    const int idx = ((n * C + c) * H + h) * W + w;
    const scalar_t x_val = x[idx];
    y[idx] = (x_val - m) * inv_std * weight[c] + bias[c];
}

torch::Tensor group_norm_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int64_t num_groups,
    double eps) {

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int channels_per_group = C / num_groups;

    auto y = torch::empty_like(x);
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto mean = torch::empty({N, num_groups}, options);
    auto var = torch::empty({N, num_groups}, options);

    // Configure thread blocks and grid for compute_stats_kernel
    dim3 threadsStats(16, 16);  // 2D thread block for spatial dimensions
    dim3 blocksStats(1, num_groups, N);  // 3D grid for batch and groups
    const int shared_mem_size = (threadsStats.x * threadsStats.y / 32) * 2 * sizeof(float);

    // Configure thread blocks and grid for normalization kernel
    dim3 threadsNorm(16, 4, 4);  // 3D thread block
    dim3 blocksNorm(
        (W + threadsNorm.x - 1) / threadsNorm.x,
        (H + threadsNorm.y - 1) / threadsNorm.y,
        (N * C + threadsNorm.z - 1) / threadsNorm.z
    );

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "group_norm_forward_cuda", ([&] {
        compute_stats_kernel<scalar_t><<<blocksStats, threadsStats, shared_mem_size, stream>>>(
            x.data_ptr<scalar_t>(),
            N, C, H, W,
            channels_per_group,
            num_groups,
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>());

        group_norm_forward_kernel<scalar_t><<<blocksNorm, threadsNorm, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            mean.data_ptr<scalar_t>(),
            var.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            bias.data_ptr<scalar_t>(),
            N, C, H, W,
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