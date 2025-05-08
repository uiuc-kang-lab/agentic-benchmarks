#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Warp-level reduction for float
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using warp reduces
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // one value per warp
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Only first warp reduces the partial sums
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// CUDA kernel with aligned, vectorized memory accesses for coalescing
__global__ void instance_norm_kernel_aligned_coalesce(
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
    // Each block processes one (N, C) instance
    int instance = blockIdx.x;
    if (instance >= N * C) return;

    int n = instance / C;
    int c = instance % C;
    int HW = H * W;

    // Pointers to the start of the current instance
    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;

    // Process data in vectorized chunks (float4).
    int num_vec = HW / 4;    // number of float4 elements
    int rem = HW % 4;        // remaining elements

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;

    // Use reinterpret_cast to load 4 floats at a time (ensuring alignment).
    const float4* x_vec = reinterpret_cast<const float4*>(x_instance);
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 v = x_vec[i];
        local_sum += v.x + v.y + v.z + v.w;
        local_sum_sq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Process any leftover elements
    int base = num_vec * 4;
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_instance[base + i];
        local_sum += v;
        local_sum_sq += v * v;
    }

    // Reduce the sums across the block
    local_sum = blockReduceSum(local_sum);
    local_sum_sq = blockReduceSum(local_sum_sq);

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float mean = local_sum / HW;
        float var = (local_sum_sq / HW) - mean * mean;
        var = (var < 0.f) ? 0.f : var;
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // Preload scale and bias (if provided)
    float scale = (weight != nullptr) ? weight[c] : 1.0f;
    float shift = (bias != nullptr)   ? bias[c]   : 0.0f;

    // Second pass: Normalize using vectorized operations
    float4* y_vec = reinterpret_cast<float4*>(y_instance);
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 v = x_vec[i];
        float4 norm;
        norm.x = ((v.x - mean) * inv_std) * scale + shift;
        norm.y = ((v.y - mean) * inv_std) * scale + shift;
        norm.z = ((v.z - mean) * inv_std) * scale + shift;
        norm.w = ((v.w - mean) * inv_std) * scale + shift;
        y_vec[i] = norm;
    }

    // Handle any remaining elements
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        int idx = num_vec * 4 + i;
        float v = x_instance[idx];
        y_instance[idx] = ((v - mean) * inv_std) * scale + shift;
    }
}

// Forward function callable from Python
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    if (weight.defined() && weight.numel() > 0)
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined() && bias.numel() > 0)
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
    
    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");
    
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];
    
    auto y = torch::empty_like(x);
    int total_instances = N * C;
    int threads = 256;

    // Launch kernel: one block per instance
    instance_norm_kernel_aligned_coalesce<<<total_instances, threads>>>(
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
    m.def("forward", &forward, "Instance Normalization forward (CUDA) with aligned memory coalescing");
}
