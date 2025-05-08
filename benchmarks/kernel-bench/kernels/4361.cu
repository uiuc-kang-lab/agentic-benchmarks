#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];  // assume blockDim.x is always a multiple of warpSize
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;
    
    val = warpReduceSum(val);
    
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    float result = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.f;

    if (wid == 0) result = warpReduceSum(result);
    return result;
}

__global__ void instance_norm_kernel_shared(
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
    const int HW = H * W;
    const int instance_idx = blockIdx.x;
    const int n = instance_idx / C;
    const int c = instance_idx % C;
    
    if (instance_idx >= N * C) return;

    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;

    float sum = 0.0f;
    float sum_sq = 0.0f;
    int tid = threadIdx.x;
    
    // Accumulate sum and sum of squares
    for (int i = tid; i < HW; i += blockDim.x) {
        float v = x_instance[i];
        sum += v;
        sum_sq += v * v;
    }
    
    // Parallel reduction to compute mean and variance
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_mean, s_inv_std;
    if (tid == 0) {
        float mean = sum / HW;
        float var = fmaxf(sum_sq / HW - mean * mean, 0.f);
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    const float mean = s_mean;
    const float inv_std = s_inv_std;
    const float w = weight ? weight[c] : 1.0f;
    const float b = bias ? bias[c] : 0.0f;
    
    // Normalize
    for (int i = tid; i < HW; i += blockDim.x) {
        float v = x_instance[i];
        y_instance[i] = (v - mean) * inv_std * w + b;
    }
}

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
    TORCH_CHECK(sizes.size() == 4, "Input must be 4D");
    
    const int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    auto y = torch::empty_like(x);
    
    const dim3 threads(256);
    const dim3 blocks(N * C);
    
    instance_norm_kernel_shared<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        weight.defined() ? weight.data_ptr<float>() : nullptr,
        bias.defined() ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );
    
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA)");
}
