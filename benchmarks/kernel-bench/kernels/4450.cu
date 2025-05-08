#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level reduction using shuffle
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction with a single __syncthreads()
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x & (warpSize - 1);
    int wid  = threadIdx.x >> 5;  // divide by warpSize
    
    // Each warp performs a reduction
    val = warpReduceSum(val);
    // Write reduced value of each warp to shared memory
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();  // Wait for all warp leaders

    // Let the first warp load all values from shared memory and reduce them
    float sum = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (threadIdx.x < warpSize) {
        sum = warpReduceSum(sum);
    }
    return sum;
}

// Kernel with minimal synchronization calls for instance normalization
__global__ void instance_norm_kernel_min_sync(
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
    const int vector_size = 4;

    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    float sum_val = 0.0f, sum_sq_val = 0.0f;
    int aligned_HW = (HW / vector_size) * vector_size;

    // Vectorized accumulation for mean and variance
    for (int i = threadIdx.x * vector_size; i < aligned_HW; i += blockDim.x * vector_size) {
        float4 vec = reinterpret_cast<const float4*>(x_ptr)[i / vector_size];
        sum_val   += vec.x + vec.y + vec.z + vec.w;
        sum_sq_val += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
    }

    // Process any remaining scalar elements
    for (int i = aligned_HW + threadIdx.x; i < HW; i += blockDim.x) {
        float v = x_ptr[i];
        sum_val += v;
        sum_sq_val += v * v;
    }

    // Reduce sums across the block
    sum_val = blockReduceSum(sum_val);
    sum_sq_val = blockReduceSum(sum_sq_val);

    // Compute mean and inverse standard deviation once per instance
    __shared__ float s_mean;
    __shared__ float s_invStd;
    if (threadIdx.x == 0) {
        float mean = sum_val / HW;
        float var = fmaxf(sum_sq_val / HW - mean * mean, 0.0f);
        s_mean = mean;
        s_invStd = rsqrtf(var + eps);
    }
    __syncthreads(); // Single sync to ensure all threads see the computed s_mean and s_invStd

    float gamma = weight ? weight[c] : 1.0f;
    float beta  = bias ? bias[c] : 0.0f;

    // Normalize using vectorized operations
    for (int i = threadIdx.x * vector_size; i < aligned_HW; i += blockDim.x * vector_size) {
        float4 vec = reinterpret_cast<const float4*>(x_ptr)[i / vector_size];
        vec.x = (vec.x - s_mean) * s_invStd * gamma + beta;
        vec.y = (vec.y - s_mean) * s_invStd * gamma + beta;
        vec.z = (vec.z - s_mean) * s_invStd * gamma + beta;
        vec.w = (vec.w - s_mean) * s_invStd * gamma + beta;
        reinterpret_cast<float4*>(y_ptr)[i / vector_size] = vec;
    }
    // Process remaining elements
    for (int i = aligned_HW + threadIdx.x; i < HW; i += blockDim.x) {
        float v = x_ptr[i];
        y_ptr[i] = (v - s_mean) * s_invStd * gamma + beta;
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
    TORCH_CHECK(x.dim() == 4, "Input tensor must be 4D (N, C, H, W)");

    int N = x.size(0);
    int C = x.size(1);
    int H = x.size(2);
    int W = x.size(3);
    auto y = torch::empty_like(x);

    int threads = 128;
    int blocks = N * C;  // one block per instance (N, C pair)

    instance_norm_kernel_min_sync<<<blocks, threads>>>(
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
    m.def("forward", &forward, "InstanceNorm minimal sync (CUDA)");
}
