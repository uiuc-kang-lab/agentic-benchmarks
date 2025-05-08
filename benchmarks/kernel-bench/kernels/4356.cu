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
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0)
        shared[wid] = val;
    __syncthreads();

    // Let first warp reduce the sums
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// CUDA kernel with vectorized loads/stores to ensure memory coalescing
__global__ void instance_norm_kernel_vec_coalesced(
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
    int instance_id = blockIdx.x;
    if (instance_id >= N * C) return;

    int n = instance_id / C;
    int c = instance_id % C;
    int HW = H * W;

    // Pointer to the beginning of this instance
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    // Process data in vectorized manner using float4
    int vec_size = HW / 4;  // number of float4 elements
    int remainder = HW % 4; // remaining elements

    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float4* x_vec = reinterpret_cast<const float4*>(x_ptr);

    // First pass: compute sum and sum of squares using vectorized loads
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 vals = x_vec[i];
        sum += vals.x + vals.y + vals.z + vals.w;
        sum_sq += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
    }
    
    int offset = vec_size * 4;
    // Process any remaining values
    for (int i = threadIdx.x; i < remainder; i += blockDim.x) {
        float v = x_ptr[offset + i];
        sum += v;
        sum_sq += v * v;
    }
    
    // Reduce sum within block
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float sharedMean;
    __shared__ float sharedVar;
    if (threadIdx.x == 0) {
        float mean = sum / static_cast<float>(HW);
        float var = sum_sq / static_cast<float>(HW) - mean * mean;
        // Clamp negative variance due to precision issues
        sharedMean = mean;
        sharedVar = (var < 0.f) ? 0.f : var;
    }
    __syncthreads();

    float mean = sharedMean;
    float var = sharedVar;
    float inv_std = rsqrtf(var + eps);

    // Precompute scale and bias if provided
    float scale = (weight != nullptr) ? weight[c] : 1.0f;
    float shift = (bias != nullptr) ? bias[c] : 0.0f;

    // Second pass: normalize using vectorized loads/stores
    float4* y_vec = reinterpret_cast<float4*>(y_ptr);
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 vals = x_vec[i];
        float4 out;
        out.x = (vals.x - mean) * inv_std;
        out.y = (vals.y - mean) * inv_std;
        out.z = (vals.z - mean) * inv_std;
        out.w = (vals.w - mean) * inv_std;
        out.x = out.x * scale + shift;
        out.y = out.y * scale + shift;
        out.z = out.z * scale + shift;
        out.w = out.w * scale + shift;
        y_vec[i] = out;
    }
    
    offset = vec_size * 4;
    for (int i = threadIdx.x; i < remainder; i += blockDim.x) {
        float val = x_ptr[offset + i];
        float norm_val = (val - mean) * inv_std;
        norm_val = norm_val * scale + shift;
        y_ptr[offset + i] = norm_val;
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

    instance_norm_kernel_vec_coalesced<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Instance Normalization forward (CUDA) with vectorized memory coalescing");
}
