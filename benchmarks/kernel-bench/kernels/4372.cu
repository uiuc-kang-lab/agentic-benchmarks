#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Warp-level reduction using __shfl_down_sync
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// CUDA kernel using __ldg for read-only loads and aligned 128-bit accesses via float4
__global__ void instance_norm_kernel_ldg(
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
    int instance_id = blockIdx.x; // one block per instance (n, c) pair
    if (instance_id >= N * C) return;

    int n = instance_id / C;
    int c = instance_id % C;
    int HW = H * W;

    // Set up pointers for the current instance
    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;

    // Process using vectorized loads/stores: 128-bit aligned using float4
    int vecCount = HW / 4; // number of float4 elements
    int rem = HW % 4;      // remaining elements

    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float4* x_vec = reinterpret_cast<const float4*>(x_instance);

    // First pass: compute sum and sum of squares using __ldg for read-only loads
    for (int i = threadIdx.x; i < vecCount; i += blockDim.x) {
        // __ldg() loads data through the read-only cache
        float4 data = __ldg(&x_vec[i]);
        sum += data.x + data.y + data.z + data.w;
        sum_sq += data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
    }

    int offset = vecCount * 4;
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float data = __ldg(&x_instance[offset + i]);
        sum += data;
        sum_sq += data * data;
    }

    // Reduce within the block
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum / HW;
        float var = sum_sq / HW - mean * mean;
        var = (var < 0.0f) ? 0.0f : var;
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;
    float scale = (weight != nullptr) ? weight[c] : 1.0f;
    float bias_val = (bias != nullptr) ? bias[c] : 0.0f;

    // Second pass: normalize using vectorized operations and __ldg() for input loads
    float4* y_vec = reinterpret_cast<float4*>(y_instance);
    for (int i = threadIdx.x; i < vecCount; i += blockDim.x) {
        float4 data = __ldg(&x_vec[i]);
        float4 norm;
        norm.x = ((data.x - mean) * inv_std * scale) + bias_val;
        norm.y = ((data.y - mean) * inv_std * scale) + bias_val;
        norm.z = ((data.z - mean) * inv_std * scale) + bias_val;
        norm.w = ((data.w - mean) * inv_std * scale) + bias_val;
        y_vec[i] = norm;
    }
    
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        int idx = offset + i;
        float data = __ldg(&x_instance[idx]);
        y_instance[idx] = ((data - mean) * inv_std * scale) + bias_val;
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
    if (weight.defined()) TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");
    if (bias.defined()) TORCH_CHECK(bias.is_cuda(), "bias must be a CUDA tensor");

    auto sizes = x.sizes();
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    auto y = torch::empty_like(x);
    int threads = 256;
    int blocks = N * C;
    instance_norm_kernel_ldg<<<blocks, threads>>>(
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
    m.def("forward", &forward, "InstanceNormalization forward (CUDA) with __ldg and aligned accesses");
}
