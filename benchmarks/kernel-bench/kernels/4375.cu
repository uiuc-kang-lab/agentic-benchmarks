#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Maximum number of channels allowed in constant memory
// Adjust MAX_CHANNELS as needed to accommodate your model
#define MAX_CHANNELS 4096

// Declare constant memory for weight and bias
__constant__ float d_const_weight[MAX_CHANNELS];
__constant__ float d_const_bias[MAX_CHANNELS];

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
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    val = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// CUDA kernel for Instance Normalization using constant memory for weight and bias
__global__ void instance_norm_kernel_const(
    const float* __restrict__ x,
    float* __restrict__ y,
    int N,
    int C,
    int H,
    int W,
    int has_weight,
    int has_bias,
    float eps
) {
    // Each block handles one (N, C) instance
    int instance_idx = blockIdx.x;
    if (instance_idx >= N * C) return;

    int n = instance_idx / C;
    int c = instance_idx % C;
    int HW = H * W;

    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    // Process data with vectorized loads using float4
    int vecCount = HW / 4;    // number of float4 elements
    int rem = HW % 4;         // remaining elements

    float sum = 0.0f, sum_sq = 0.0f;
    const float4* x_vec = reinterpret_cast<const float4*>(x_ptr);

    // First pass: vectorized reduction for sum and sum_sq
    for (int i = threadIdx.x; i < vecCount; i += blockDim.x) {
        float4 vals = x_vec[i];
        sum    += vals.x + vals.y + vals.z + vals.w;
        sum_sq += vals.x * vals.x + vals.y * vals.y + vals.z * vals.z + vals.w * vals.w;
    }

    int offset = vecCount * 4;
    // Process remaining elements
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_ptr[offset + i];
        sum    += v;
        sum_sq += v * v;
    }

    // Reduce sums within the block
    sum    = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum / HW;
        float var  = sum_sq / HW - mean * mean;
        var = (var < 0.f) ? 0.f : var;
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;

    // Load weight and bias from constant memory if provided
    float scale = has_weight ? d_const_weight[c] : 1.0f;
    float shift = has_bias   ? d_const_bias[c]   : 0.0f;

    // Second pass: normalize using vectorized operations
    float4* y_vec = reinterpret_cast<float4*>(y_ptr);
    for (int i = threadIdx.x; i < vecCount; i += blockDim.x) {
        float4 vals = x_vec[i];
        float4 out;
        out.x = ((vals.x - mean) * inv_std * scale) + shift;
        out.y = ((vals.y - mean) * inv_std * scale) + shift;
        out.z = ((vals.z - mean) * inv_std * scale) + shift;
        out.w = ((vals.w - mean) * inv_std * scale) + shift;
        y_vec[i] = out;
    }

    // Handle remaining elements
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        int idx = vecCount * 4 + i;
        float v = x_ptr[idx];
        y_ptr[idx] = ((v - mean) * inv_std * scale) + shift;
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
    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    auto y = torch::empty_like(x);

    int has_weight = 0;
    int has_bias = 0;
    if (weight.defined() && weight.numel() > 0) {
        TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
        TORCH_CHECK(weight.numel() >= C, "weight size must be at least C");
        has_weight = 1;
        // Copy weight data to constant memory
        cudaMemcpyToSymbol(d_const_weight, weight.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    }
    if (bias.defined() && bias.numel() > 0) {
        TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");
        TORCH_CHECK(bias.numel() >= C, "bias size must be at least C");
        has_bias = 1;
        // Copy bias data to constant memory
        cudaMemcpyToSymbol(d_const_bias, bias.data_ptr<float>(), C * sizeof(float), 0, cudaMemcpyDeviceToDevice);
    }

    int threads = 256;
    int blocks = N * C;

    instance_norm_kernel_const<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N, C, H, W,
        has_weight, has_bias,
        static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA) with constant memory");
}
