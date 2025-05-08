#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    
    return val;
}

__global__ void instance_norm_kernel(
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
    const int instance_idx = blockIdx.x;
    const int n = instance_idx / C;
    const int c = instance_idx % C;
    const int HW = H * W;
    
    if (instance_idx >= N * C) return;

    // Align starting points to warp boundaries
    const int warp_size = 32;
    const int aligned_hw = ((HW + warp_size - 1) / warp_size) * warp_size;
    
    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;
    
    // First pass: compute mean and variance
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    // Process elements in warp-aligned chunks
    #pragma unroll 4
    for (int i = threadIdx.x; i < aligned_hw; i += blockDim.x) {
        float val = (i < HW) ? x_instance[i] : 0.0f;
        sum += val;
        sum_sq += val * val;
    }
    
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);
    
    __shared__ float s_mean, s_var;
    
    if (threadIdx.x == 0) {
        s_mean = sum / HW;
        s_var = (sum_sq / HW) - (s_mean * s_mean);
        s_var = max(s_var, 0.0f);
    }
    __syncthreads();
    
    // Pre-compute normalization terms
    const float mean = s_mean;
    const float inv_std = rsqrtf(s_var + eps);
    const float w = weight ? weight[c] : 1.0f;
    const float b = bias ? bias[c] : 0.0f;
    
    // Second pass: normalize with aligned access
    #pragma unroll 4
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float val = x_instance[i];
        float norm_val = (val - mean) * inv_std;
        y_instance[i] = fmaf(norm_val, w, b);
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

    int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    auto y = torch::empty_like(x);

    dim3 threads(256);
    dim3 blocks(N * C);

    instance_norm_kernel<<<blocks, threads>>>(
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