#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Structure to hold pair of sum and sumsq
struct Pair {
    float sum;
    float sum_sq;
};

// Warp-level reduction for Pair
__inline__ __device__ Pair warpReducePair(Pair val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val.sum    += __shfl_down_sync(0xffffffff, val.sum, offset);
        val.sum_sq += __shfl_down_sync(0xffffffff, val.sum_sq, offset);
    }
    return val;
}

// Block-level reduction for Pair with a single synchronization barrier
__inline__ __device__ Pair blockReducePair(Pair val) {
    __shared__ Pair shared[32];  // one per warp
    int lane = threadIdx.x & (warpSize - 1);
    int wid = threadIdx.x / warpSize;

    // Each warp reduces its own value
    val = warpReducePair(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();  // Necessary for shared memory consistency across warps

    // First warp loads the partial sums
    Pair result;
    if (threadIdx.x < blockDim.x / warpSize) {
        result = shared[lane];
    } else {
        result.sum = 0.0f;
        result.sum_sq = 0.0f;
    }
    if (wid == 0) {
        result = warpReducePair(result);
    }
    return result;
}

// CUDA kernel for Instance Normalization with minimal __syncthreads()
__global__ void instance_norm_kernel(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N,
    int C,
    int H,
    int W,
    float eps
) {
    int instance_id = blockIdx.x;
    if (instance_id >= N * C) return;

    int n = instance_id / C;
    int c = instance_id % C;
    int HW = H * W;

    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;

    // Accumulate sum and sum of squares in a single pass
    Pair thread_pair;
    thread_pair.sum = 0.0f;
    thread_pair.sum_sq = 0.0f;

    int vec_count = HW / 4;
    int rem = HW % 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x_instance);

    // Process data with vectorized loads
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        float4 data = x_vec[i];
        float local_sum = data.x + data.y + data.z + data.w;
        float local_sum_sq = data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
        thread_pair.sum += local_sum;
        thread_pair.sum_sq += local_sum_sq;
    }

    int offset = vec_count * 4;
    // Process remaining elements
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_instance[offset + i];
        thread_pair.sum += v;
        thread_pair.sum_sq += v * v;
    }

    // Reduce the pair across the block with only one __syncthreads() inside blockReducePair
    Pair agg = blockReducePair(thread_pair);

    // All threads now have the same aggregated result
    float mean = agg.sum / HW;
    float var = agg.sum_sq / HW - mean * mean;
    var = (var < 0.0f) ? 0.0f : var;
    float inv_std = rsqrtf(var + eps);

    // Precompute scale and bias if provided
    float w_val = weight ? weight[c] : 1.0f;
    float b_val = bias ? bias[c] : 0.0f;

    // Normalize the input using vectorized operations
    float4* y_vec = reinterpret_cast<float4*>(y_instance);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        float4 data = x_vec[i];
        data.x = ((data.x - mean) * inv_std) * w_val + b_val;
        data.y = ((data.y - mean) * inv_std) * w_val + b_val;
        data.z = ((data.z - mean) * inv_std) * w_val + b_val;
        data.w = ((data.w - mean) * inv_std) * w_val + b_val;
        y_vec[i] = data;
    }

    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_instance[offset + i];
        y_instance[offset + i] = ((v - mean) * inv_std) * w_val + b_val;
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
    if (weight.defined()) {
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    }
    if (bias.defined()) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    }

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");

    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];
    auto y = torch::empty_like(x);

    int threads = 256;
    int blocks = N * C;
    instance_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA) with minimal synchronizations");
}
