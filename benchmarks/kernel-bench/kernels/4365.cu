#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Warp reduce sum for float
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

// Block reduce sum using warp reductions
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // one value per warp
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    __syncthreads();

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Use only the first warp to complete the reduction
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// Combined Instance Norm kernel: Uses vectorized memory access, preloaded per-channel parameters, and CUDA streams
__global__ void instance_norm_kernel_combined(
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
    int instance_id = blockIdx.x;  // each block handles one (N, C) instance
    if (instance_id >= N * C) return;

    int n = instance_id / C;
    int c = instance_id % C;
    int HW = H * W;

    // Pointers to the beginning of this instance
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    // Use vectorized memory operations if possible
    int num_vec = HW / 4;  // number of float4 elements
    int rem = HW % 4;      // remaining elements

    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;
    const float4* x_vec = reinterpret_cast<const float4*>(x_ptr);

    // First pass: compute partial sums and squared sums using vectorized loads
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 data = x_vec[i];
        sum_val += data.x + data.y + data.z + data.w;
        sum_sq_val += data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
    }

    int offset = num_vec * 4;
    // Process any remaining elements
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_ptr[offset + i];
        sum_val += v;
        sum_sq_val += v * v;
    }

    // Reduce sums across the block
    sum_val = blockReduceSum(sum_val);
    sum_sq_val = blockReduceSum(sum_sq_val);

    __shared__ float sharedMean;
    __shared__ float sharedVar;
    if (threadIdx.x == 0) {
        float mean = sum_val / static_cast<float>(HW);
        float var = sum_sq_val / static_cast<float>(HW) - mean * mean;
        // Clamp negative variance (possible due to numerical precision)
        var = (var < 0.f) ? 0.f : var;
        sharedMean = mean;
        sharedVar = var;
    }
    __syncthreads();

    float mean = sharedMean;
    float var = sharedVar;
    float inv_std = rsqrtf(var + eps);

    // Preload channel-specific weight and bias to registers (if provided)
    float gamma = (weight != nullptr) ? weight[c] : 1.0f;
    float beta  = (bias != nullptr)  ? bias[c]  : 0.0f;

    // Second pass: apply normalization using vectorized operations
    float4* y_vec = reinterpret_cast<float4*>(y_ptr);
    for (int i = threadIdx.x; i < num_vec; i += blockDim.x) {
        float4 data = x_vec[i];
        float t0 = (data.x - mean) * inv_std;
        float t1 = (data.y - mean) * inv_std;
        float t2 = (data.z - mean) * inv_std;
        float t3 = (data.w - mean) * inv_std;
        // Apply scale (gamma) and shift (beta)
        t0 = t0 * gamma + beta;
        t1 = t1 * gamma + beta;
        t2 = t2 * gamma + beta;
        t3 = t3 * gamma + beta;
        float4 norm_data = {t0, t1, t2, t3};
        y_vec[i] = norm_data;
    }

    offset = num_vec * 4;
    // Process remaining elements
    for (int i = threadIdx.x; i < rem; i += blockDim.x) {
        float v = x_ptr[offset + i];
        float norm_val = (v - mean) * inv_std;
        norm_val = norm_val * gamma + beta;
        y_ptr[offset + i] = norm_val;
    }
}

// Forward function called from Python with CUDA stream for asynchronous execution
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

    // Launch kernel: one block per (N, C) instance
    int threads = 256;
    int blocks = N * C;

    // Create and use a CUDA stream for asynchronous execution
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    instance_norm_kernel_combined<<<blocks, threads, 0, stream>>>(
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

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Combined Instance Normalization forward (CUDA) with stream and coalesced memory accesses");
}
