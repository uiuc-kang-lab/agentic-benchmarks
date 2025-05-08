#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// Kernel 1: Compute per-instance sum and sum-of-squares using atomic adds
// Each instance (n, c) is processed by multiple blocks; each block processes a chunk of the instance's data

__global__ void instance_norm_stats_kernel(
    const float* __restrict__ x,
    float* __restrict__ sum_arr,
    float* __restrict__ sumsq_arr,
    int N, int C, int H, int W,
    int blocks_per_instance
) {
    // Total number of elements per instance
    int HW = H * W;
    // Determine the instance this block is working on
    int num_instances = N * C;
    int instance_id = blockIdx.x % num_instances;
    int block_in_instance = blockIdx.x / num_instances;

    // Divide the instance data among blocks assigned to it
    int chunk_size = (HW + blocks_per_instance - 1) / blocks_per_instance;
    int start = block_in_instance * chunk_size;
    int end = min(start + chunk_size, HW);

    const float* x_ptr = x + instance_id * HW;

    // Each thread processes a portion of the assigned chunk
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float val = x_ptr[i];
        local_sum += val;
        local_sumsq += val * val;
    }

    // Block-level reduction in shared memory
    extern __shared__ float shared[]; // layout: [0: blockDim.x] -> sum, [blockDim.x: 2*blockDim.x] -> sumsq
    float* shared_sum = shared;
    float* shared_sumsq = shared + blockDim.x;
    shared_sum[threadIdx.x] = local_sum;
    shared_sumsq[threadIdx.x] = local_sumsq;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
            shared_sumsq[threadIdx.x] += shared_sumsq[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Use atomic operations just once per block to update the global sum arrays
    if (threadIdx.x == 0) {
        atomicAdd(&sum_arr[instance_id], shared_sum[0]);
        atomicAdd(&sumsq_arr[instance_id], shared_sumsq[0]);
    }
}

// Kernel 2: Apply normalization using the computed statistics
__global__ void instance_norm_apply_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ sum_arr,
    const float* __restrict__ sumsq_arr,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int H, int W,
    float eps
) {
    int instance_id = blockIdx.x;  // one block per instance (n, c)
    int HW = H * W;

    // Retrieve precomputed sums for this instance
    float sum = sum_arr[instance_id];
    float sumsq = sumsq_arr[instance_id];
    float mean = sum / HW;
    float var = sumsq / HW - mean * mean;
    if (var < 0.f) var = 0.f;
    float invstd = rsqrtf(var + eps);

    int n = instance_id / C;
    int c = instance_id % C;
    float scale = (weight != nullptr) ? weight[c] : 1.0f;
    float shift = (bias != nullptr) ? bias[c] : 0.0f;

    const float* x_ptr = x + instance_id * HW;
    float* y_ptr = y + instance_id * HW;

    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float val = x_ptr[i];
        y_ptr[i] = ((val - mean) * invstd) * scale + shift;
    }
}

// Forward function: Launch two kernels -- one for stats accumulation and one for normalization

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W)");
    
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    int num_instances = N * C;
    int HW = H * W;

    auto y = torch::empty_like(x);

    // Allocate temporary buffers for per-instance sum and sumsq; initialize with zeros
    auto options = torch::TensorOptions().device(x.device()).dtype(x.dtype());
    auto sum_arr = torch::zeros({num_instances}, options);
    auto sumsq_arr = torch::zeros({num_instances}, options);

    // Heuristic: Use multiple blocks per instance when HW is large, otherwise one block per instance
    int blocks_per_instance = (HW > 1024) ? 4 : 1;
    int total_blocks = num_instances * blocks_per_instance;
    int threads = 256;
    int shared_mem_size = threads * 2 * sizeof(float); // for two shared arrays

    // Launch Kernel 1: Compute statistics with minimal atomic usage in global memory
    instance_norm_stats_kernel<<<total_blocks, threads, shared_mem_size>>>(
        x.data_ptr<float>(),
        sum_arr.data_ptr<float>(),
        sumsq_arr.data_ptr<float>(),
        N, C, H, W,
        blocks_per_instance
    );

    // Launch Kernel 2: Normalize each instance using the computed mean and variance
    int threads_norm = 256;
    instance_norm_apply_kernel<<<num_instances, threads_norm>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        sum_arr.data_ptr<float>(),
        sumsq_arr.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Norm forward with atomic optimizations (CUDA)");
}
