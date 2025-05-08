#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Warp level reduction for a single float
__inline__ __device__ float warpReduceSum(float val) {
    // Use full mask for current warp
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block level reduction using warpReduceSum
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // one per warp
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // First warp loads results from each warp
    val = (threadIdx.x < blockDim.x/warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Optimized instance normalization kernel
// Computes mean and variance in a vectorized first pass and then normalizes input in a second pass
// Combines the best ideas from two previous kernels
__global__ void instance_norm_kernel_optimized(
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
    // Each block processes one (N, C) instance
    const int instance_idx = blockIdx.x;
    if (instance_idx >= N * C) return;

    const int n = instance_idx / C;
    const int c = instance_idx % C;
    const int HW = H * W;

    // Set up pointers for the current instance
    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;

    // Use vectorized memory access (4 floats at a time) if possible
    const int num_vec = HW / 4;  // number of float4 elements
    const int rem = HW % 4;      // remaining elements

    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;

    const float4* x_vec = reinterpret_cast<const float4*>(x_instance);

    // First pass: reduction to compute sum and squared sum
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 data = x_vec[i];
        sum_val    += data.x + data.y + data.z + data.w;
        sum_sq_val += data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
    }

    int offset = num_vec * 4;
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_instance[offset + i];
        sum_val    += v;
        sum_sq_val += v * v;
    }

    // Reduce sums across the block
    sum_val    = blockReduceSum(sum_val);
    sum_sq_val = blockReduceSum(sum_sq_val);

    // Shared memory to store the computed mean and inverse std
    __shared__ float s_mean;
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum_val / HW;
        float var  = sum_sq_val / HW - mean * mean;
        var = (var < 0.f) ? 0.f : var;
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    const float mean = s_mean;
    const float inv_std = s_inv_std;

    // Precompute weight and bias values if provided
    const float w = (weight != nullptr) ? weight[c] : 1.0f;
    const float b = (bias != nullptr)   ? bias[c]   : 0.0f;

    // Second pass: normalize the input using vectorized stores
    float4* y_vec = reinterpret_cast<float4*>(y_instance);
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 data = x_vec[i];
        float4 norm;
        norm.x = ((data.x - mean) * inv_std) * w + b;
        norm.y = ((data.y - mean) * inv_std) * w + b;
        norm.z = ((data.z - mean) * inv_std) * w + b;
        norm.w = ((data.w - mean) * inv_std) * w + b;
        y_vec[i] = norm;
    }

    offset = num_vec * 4;
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_instance[offset + i];
        y_instance[offset + i] = ((v - mean) * inv_std) * w + b;
    }
}

// Forward function callable from Python
// Uses a CUDA stream for asynchronous execution
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

    // One block per (N, C) instance
    int blocks = N * C;
    int threads = 256;

    // Create a CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    instance_norm_kernel_optimized<<<blocks, threads, 0, stream>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N, C, H, W,
        static_cast<float>(eps)
    );

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Instance Normalization forward (CUDA)");
}
