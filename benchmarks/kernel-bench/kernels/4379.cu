#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Warp-level reduction using shfl_down_sync, no shared memory required
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel: one block per (N, C) instance, with blockDim.x = warpSize (32 threads)
// This avoids shared memory use for reduction by using warp-level primitives only.
__global__ void instance_norm_kernel_warponly(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const int N,
    const int C,
    const int H,
    const int W,
    const float eps
) {
    // Each block processes one instance (i.e. one (N, C) pair)
    int instance = blockIdx.x;
    if (instance >= N * C) return;

    int n = instance / C;
    int c = instance % C;
    int HW = H * W;

    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;

    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // Process data in 128-bit chunks with float4 (vectorized load)
    int vecCount = HW / 4;
    int rem = HW % 4;
    const float4* x_vec = reinterpret_cast<const float4*>(x_instance);

    // First pass: accumulate sum and sum of squares
    for (int i = threadIdx.x; i < vecCount; i += warpSize) {
        float4 data = __ldg(&x_vec[i]);
        local_sum += data.x + data.y + data.z + data.w;
        local_sq_sum += data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
    }
    
    int offset = vecCount * 4;
    for (int i = threadIdx.x; i < rem; i += warpSize) {
        float val = __ldg(&x_instance[offset + i]);
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Warp-level reduction: as blockDim.x == warpSize, all threads are in the same warp
    local_sum = warpReduceSum(local_sum);
    local_sq_sum = warpReduceSum(local_sq_sum);

    int lane = threadIdx.x; // all threads in a warp
    float mean, inv_std;
    if (lane == 0) {
        float m = local_sum / HW;
        float var = local_sq_sum / HW - m * m;
        var = (var < 0.0f) ? 0.0f : var;
        mean = m;
        inv_std = rsqrtf(var + eps);
    }
    // Broadcast the computed mean and inv_std to all threads in the warp
    mean = __shfl_sync(0xffffffff, mean, 0);
    inv_std = __shfl_sync(0xffffffff, inv_std, 0);

    // Load scale and shift parameters via read-only cache
    float scal = (weight != nullptr) ? __ldg(&weight[c]) : 1.0f;
    float shft = (bias != nullptr)   ? __ldg(&bias[c])   : 0.0f;

    // Second pass: apply normalization using the computed mean and inv_std
    for (int i = threadIdx.x; i < vecCount; i += warpSize) {
        float4 data = __ldg(&x_vec[i]);
        float4 result;
        result.x = ((data.x - mean) * inv_std * scal) + shft;
        result.y = ((data.y - mean) * inv_std * scal) + shft;
        result.z = ((data.z - mean) * inv_std * scal) + shft;
        result.w = ((data.w - mean) * inv_std * scal) + shft;
        reinterpret_cast<float4*>(y_instance)[i] = result;
    }
    for (int i = threadIdx.x; i < rem; i += warpSize) {
        int idx = offset + i;
        float val = x_instance[idx];
        y_instance[idx] = ((val - mean) * inv_std * scal) + shft;
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
    
    // Launch one block per instance using 32 threads (one warp per instance)
    int threads = 32;
    int blocks = N * C;
    instance_norm_kernel_warponly<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Instance Normalization forward (CUDA) using warp-level primitives only");
}
