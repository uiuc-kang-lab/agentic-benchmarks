#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    __syncthreads();

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
    int N,
    int C,
    int H,
    int W,
    float eps
) {
    const int instance_idx = blockIdx.x;
    const int chunk_idx = blockIdx.y;
    
    if (instance_idx >= N * C) return;
    
    const int n = instance_idx / C;
    const int c = instance_idx % C;
    const int HW = H * W;
    const int chunk_size = (HW + gridDim.y - 1) / gridDim.y;
    const int chunk_start = chunk_idx * chunk_size;
    const int chunk_end = min(chunk_start + chunk_size, HW);
    
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;
    
    __shared__ float s_mean, s_var;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int idx = chunk_start + threadIdx.x; idx < chunk_end; idx += blockDim.x) {
        float val = x_ptr[idx];
        sum += val;
        sum_sq += val * val;
    }
    
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);
    
    if (threadIdx.x == 0) {
        atomicAdd(&s_mean, sum);
        atomicAdd(&s_var, sum_sq);
    }
    __syncthreads();
    
    if (chunk_idx == 0 && threadIdx.x == 0) {
        s_mean = s_mean / HW;
        s_var = s_var / HW - s_mean * s_mean;
        s_var = (s_var < 0.f) ? 0.f : s_var;
    }
    __syncthreads();
    
    const float mean = s_mean;
    const float var = s_var;
    const float scale = weight ? weight[c] : 1.0f;
    const float shift = bias ? bias[c] : 0.0f;
    
    for (int idx = chunk_start + threadIdx.x; idx < chunk_end; idx += blockDim.x) {
        float val = x_ptr[idx];
        float norm_val = (val - mean) / sqrtf(var + eps);
        y_ptr[idx] = norm_val * scale + shift;
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
    TORCH_CHECK(sizes.size() == 4, "Input tensor must be 4D: (N, C, H, W)");

    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    auto y = torch::empty_like(x);

    dim3 threads(256);
    dim3 blocks(N * C, 8); // Use 8 chunks for H*W dimension
    
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