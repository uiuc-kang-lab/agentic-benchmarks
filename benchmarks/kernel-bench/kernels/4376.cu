#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Kernel 1: Reduction kernel that computes the sum and sum of squares for each instance
// by partitioning the HW elements among multiple blocks per instance. Each block computes
// its partial sums and then uses atomicAdd to accumulate to a global temporary buffer.

__global__ void instance_norm_reduce_kernel(
    const float* __restrict__ x,
    float* __restrict__ sum, 
    float* __restrict__ sum_sq,
    int N, int C, int H, int W,
    int blocksPerInstance
) {
    // Determine which instance this block is processing
    int instance_id = blockIdx.x / blocksPerInstance;
    if (instance_id >= N * C) return;

    int block_id_in_instance = blockIdx.x % blocksPerInstance;
    int HW = H * W;
    // Calculate slice boundaries for this block within the instance
    int slice = (HW + blocksPerInstance - 1) / blocksPerInstance; // ceiling division
    int start = block_id_in_instance * slice;
    int end = start + slice;
    if (end > HW) end = HW;

    const float* x_instance = x + instance_id * HW;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Each thread processes a portion of the assigned slice
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float val = x_instance[i];
        local_sum += val;
        local_sum_sq += val * val;
    }

    // Use shared memory to reduce within the block
    extern __shared__ float shared[];  // Shared memory: first half for sum, second half for sum_sq
    float* shared_sum = shared;
    float* shared_sum_sq = shared + blockDim.x;

    int tid = threadIdx.x;
    shared_sum[tid] = local_sum;
    shared_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sum_sq[tid] += shared_sum_sq[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 atomically adds the block's partial results to the global accumulators
    if (tid == 0) {
        atomicAdd(&sum[instance_id], shared_sum[0]);
        atomicAdd(&sum_sq[instance_id], shared_sum_sq[0]);
    }
}

// Kernel 2: Normalization kernel that uses the computed sum and sum_sq for each instance
// to compute the mean and inverse standard deviation and then normalizes each element.

__global__ void instance_norm_normalize_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ sum, 
    const float* __restrict__ sum_sq,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int H, int W,
    float eps
) {
    int instance_id = blockIdx.x;
    if (instance_id >= N * C) return;

    int HW = H * W;
    const float* x_instance = x + instance_id * HW;
    float* y_instance = y + instance_id * HW;

    // Compute mean and variance for the instance
    float mean = sum[instance_id] / HW;
    float var = sum_sq[instance_id] / HW - mean * mean;
    if (var < 0.f) var = 0.f;
    float inv_std = rsqrtf(var + eps);

    // Determine channel index (instance_id = n * C + c)
    int channel = instance_id % C;
    float w = (weight != nullptr) ? weight[channel] : 1.0f;
    float b = (bias != nullptr)   ? bias[channel]   : 0.0f;

    // Each thread normalizes a subset of the HW elements
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float val = x_instance[i];
        y_instance[i] = ((val - mean) * inv_std * w) + b;
    }
}

// Forward function called from Python

torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    if (weight.defined() && weight.numel() > 0) {
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    }
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    }

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];
    int instance_count = N * C;
    int HW = H * W;

    // Create output tensor
    torch::Tensor y = torch::empty_like(x);

    // Allocate temporary buffers for the sums
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor sum = torch::zeros({instance_count}, options);
    torch::Tensor sum_sq = torch::zeros({instance_count}, options);

    // Determine the number of blocks per instance for the reduction kernel
    // Here we choose a slice size of ~8192 elements per block (tunable parameter)
    int blocksPerInstance = (HW + 8191) / 8192;
    if (blocksPerInstance < 1) blocksPerInstance = 1;
    int total_blocks = instance_count * blocksPerInstance;

    int threads = 256;
    size_t shared_mem = threads * 2 * sizeof(float); // for shared sum and sum_sq arrays

    // Launch the reduction kernel to compute sums using atomicAdd
    instance_norm_reduce_kernel<<<total_blocks, threads, shared_mem>>>(
        x.data_ptr<float>(),
        sum.data_ptr<float>(),
        sum_sq.data_ptr<float>(),
        N, C, H, W, blocksPerInstance
    );

    // Launch the normalization kernel with one block per instance
    int norm_threads = 256;
    instance_norm_normalize_kernel<<<instance_count, norm_threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        sum.data_ptr<float>(),
        sum_sq.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA) with atomic reductions");
}
