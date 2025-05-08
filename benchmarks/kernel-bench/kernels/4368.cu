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
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;
    
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

// CUDA kernel with loop unrolling to reduce loop overhead
__global__ void instancenorm_unroll_kernel(
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

    // Pointers to the instance data
    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;

    int vec_elements = HW / 4;  // number of float4 elements
    int rem = HW % 4;           // remaining elements

    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;
    const float4* x_vec = reinterpret_cast<const float4*>(x_instance);

    // First pass: vectorized load with manual loop unrolling
    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_elements; i += blockDim.x) {
        float4 data = x_vec[i];
        sum_val += data.x + data.y + data.z + data.w;
        sum_sq_val += data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
    }

    int offset = vec_elements * 4;
    #pragma unroll 4
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_instance[offset + i];
        sum_val += v;
        sum_sq_val += v * v;
    }

    // Reduce sums within block
    sum_val = blockReduceSum(sum_val);
    sum_sq_val = blockReduceSum(sum_sq_val);

    __shared__ float sharedMean;
    __shared__ float sharedInvStd;
    if (threadIdx.x == 0) {
        float mean = sum_val / static_cast<float>(HW);
        float var = sum_sq_val / static_cast<float>(HW) - mean * mean;
        var = (var < 0.0f) ? 0.0f : var;
        sharedMean = mean;
        sharedInvStd = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = sharedMean;
    float inv_std = sharedInvStd;
    float w_val = (weight ? weight[c] : 1.0f);
    float b_val = (bias ? bias[c] : 0.0f);

    // Second pass: normalize using vectorized operations with unrolling
    float4* y_vec = reinterpret_cast<float4*>(y_instance);
    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_elements; i += blockDim.x) {
        float4 data = x_vec[i];
        float4 out;
        out.x = ((data.x - mean) * inv_std * w_val) + b_val;
        out.y = ((data.y - mean) * inv_std * w_val) + b_val;
        out.z = ((data.z - mean) * inv_std * w_val) + b_val;
        out.w = ((data.w - mean) * inv_std * w_val) + b_val;
        y_vec[i] = out;
    }

    offset = vec_elements * 4;
    #pragma unroll 4
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_instance[offset + i];
        y_instance[offset + i] = ((v - mean) * inv_std * w_val) + b_val;
    }
}

// Forward function exposed to Python
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
    instancenorm_unroll_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N, C, H, W, static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA) with loop unrolling");
}
