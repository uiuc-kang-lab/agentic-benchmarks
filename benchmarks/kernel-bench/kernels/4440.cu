#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Helper device function: Warp-level reduction
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

// Helper device function: Block-level reduction using warp reductions
__inline__ __device__ float blockReduceSum(float val) {
    // Shared memory for partial sums; one value per warp
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    __syncthreads();

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Only the first warp loads the partial sums
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Device function to compute instance statistics (mean and variance)
// This function uses vectorized loads to accelerate summation
__device__ void compute_instance_stats(const float* x_ptr, int HW, float* out_mean, float* out_var) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const int vector_size = 4;
    int aligned_HW = (HW / vector_size) * vector_size;

    // Vectorized loop
    int i = threadIdx.x * vector_size;
    while (i < aligned_HW) {
        // Load 4 floats at a time
        float4 vec = reinterpret_cast<const float4*>(x_ptr)[i / vector_size];
        sum     += vec.x + vec.y + vec.z + vec.w;
        sum_sq  += vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
        i += blockDim.x * vector_size;
    }

    // Process any remaining elements
    for (i = aligned_HW + threadIdx.x; i < HW; i += blockDim.x) {
        float v = x_ptr[i];
        sum += v;
        sum_sq += v * v;
    }

    // Reduce the partial results within the block
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    // Thread 0 computes final mean and variance
    if (threadIdx.x == 0) {
        *out_mean = sum / HW;
        float mean = *out_mean;
        *out_var = fmaxf(sum_sq / HW - mean * mean, 0.0f);
    }
}

// Device function to perform normalization on an instance using vectorized memory accesses
__device__ void normalize_instance(const float* x_ptr, float* y_ptr, int HW, float mean, float inv_std, float gamma, float beta) {
    const int vector_size = 4;
    int aligned_HW = (HW / vector_size) * vector_size;

    int idx = threadIdx.x * vector_size;
    while (idx < aligned_HW) {
        float4 vec = reinterpret_cast<const float4*>(x_ptr)[idx / vector_size];
        vec.x = (vec.x - mean) * inv_std * gamma + beta;
        vec.y = (vec.y - mean) * inv_std * gamma + beta;
        vec.z = (vec.z - mean) * inv_std * gamma + beta;
        vec.w = (vec.w - mean) * inv_std * gamma + beta;
        reinterpret_cast<float4*>(y_ptr)[idx / vector_size] = vec;
        idx += blockDim.x * vector_size;
    }

    // Process any remaining elements
    for (int i = aligned_HW + threadIdx.x; i < HW; i += blockDim.x) {
        float v = x_ptr[i];
        y_ptr[i] = (v - mean) * inv_std * gamma + beta;
    }
}

// Modular CUDA kernel for Instance Normalization
// Each block handles one (N, C) instance
__global__ void instance_norm_kernel_modular(
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
    if (instance_id >= N * C) return; __syncthreads();

    int n = instance_id / C;
    int c = instance_id % C;
    int HW = H * W;

    // Pointers for the current instance
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    // Shared memory to store the computed mean and variance
    __shared__ float s_mean;
    __shared__ float s_var;

    // Compute statistics (all threads in the block participate)
    compute_instance_stats(x_ptr, HW, &s_mean, &s_var);
    __syncthreads();

    float inv_std = rsqrtf(s_var + eps);
    float gamma = (weight != nullptr) ? weight[c] : 1.0f;
    float beta  = (bias != nullptr)  ? bias[c]  : 0.0f;

    // Normalize the input using the computed statistics
    normalize_instance(x_ptr, y_ptr, HW, s_mean, inv_std, gamma, beta);
}

// Forward function called from Python
// This function launches the modular instance normalization kernel
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

    // Each block processes one (N, C) instance
    int threads = 256;
    int blocks = N * C;

    instance_norm_kernel_modular<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Modular Instance Normalization forward (CUDA)");
}
