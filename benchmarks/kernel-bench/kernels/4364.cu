#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

__inline__ __device__ float warpReduceSum(float val) {
#if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ >= 9)
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
#else
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down(val, offset);
    }
#endif
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    __syncthreads();

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

__global__ void instance_norm_kernel_optimized(
    const float* __restrict__ x,
    float* __restrict__ y,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    int N, int C, int H, int W, float eps
) {
    int instance_id = blockIdx.x;
    if (instance_id >= N * C) {
        return;
    }

    int n = instance_id / C;
    int c = instance_id % C;
    int HW = H * W;

    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr       = y + (n * C + c) * HW;

    int num_vec = HW / 4;
    int rem = HW % 4;

    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;

    const float4* x_vec = reinterpret_cast<const float4*>(x_ptr);

    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 data = x_vec[i];
        sum_val += data.x + data.y + data.z + data.w;
        sum_sq_val += data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
    }

    int offset = num_vec * 4;
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_ptr[offset + i];
        sum_val += v;
        sum_sq_val += v * v;
    }

    sum_val    = blockReduceSum(sum_val);
    sum_sq_val = blockReduceSum(sum_sq_val);

    __shared__ float sharedMean;
    __shared__ float sharedVar;
    if (threadIdx.x == 0) {
        float mean = sum_val / (float)HW;
        float var  = sum_sq_val / (float)HW - mean * mean;
        var = (var < 0.f) ? 0.f : var;
        sharedMean = mean;
        sharedVar  = var;
    }
    __syncthreads();

    float mean = sharedMean;
    float var = sharedVar;
    float inv_std = rsqrtf(var + eps);

    float4* y_vec = reinterpret_cast<float4*>(y_ptr);
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 data = x_vec[i];
        float t0 = (data.x - mean) * inv_std;
        float t1 = (data.y - mean) * inv_std;
        float t2 = (data.z - mean) * inv_std;
        float t3 = (data.w - mean) * inv_std;
        if (weight && bias) {
            float w = weight[c];
            float b = bias[c];
            t0 = t0 * w + b;
            t1 = t1 * w + b;
            t2 = t2 * w + b;
            t3 = t3 * w + b;
        }
        float4 norm_data = {t0, t1, t2, t3};
        y_vec[i] = norm_data;
    }

    offset = num_vec * 4;
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_ptr[offset + i];
        float norm_val = (v - mean) * inv_std;
        if (weight && bias) {
            norm_val = norm_val * weight[c] + bias[c];
        }
        y_ptr[offset + i] = norm_val;
    }
}

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
    instance_norm_kernel_optimized<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N, C, H, W, static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA) optimized");
}
