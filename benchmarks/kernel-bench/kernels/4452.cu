#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


// Warp-level reduction
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

// Block-level reduction using warp reductions
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // One warp per 32 threads
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

// CUDA kernel using stride loops to handle workloads larger than available threads
__global__ void instance_norm_kernel_stride(
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

    // Pointers for this instance
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    // First pass: compute sum and sum of squares with a stride loop
    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float v = __ldg(x_ptr + i);
        sum_val += v;
        sum_sq_val += v * v;
    }

    sum_val = blockReduceSum(sum_val);
    sum_sq_val = blockReduceSum(sum_sq_val);

    __shared__ float sharedMean;
    __shared__ float sharedInvStd;
    if (threadIdx.x == 0) {
        float mean = sum_val / HW;
        float var = sum_sq_val / HW - mean * mean;
        var = (var < 0.f) ? 0.f : var;  // safeguard against negative variance
        sharedMean = mean;
        sharedInvStd = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = sharedMean;
    float inv_std = sharedInvStd;
    float w = (weight != nullptr) ? __ldg(weight + c) : 1.0f;
    float b = (bias != nullptr) ? __ldg(bias + c) : 0.0f;

    // Second pass: apply normalization using a stride loop
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float v = __ldg(x_ptr + i);
        y_ptr[i] = (v - mean) * inv_std * w + b;
    }
}

// Forward function exposed via Pybind11
torch::Tensor forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    double eps
) {
    TORCH_CHECK(x.is_cuda(), "Input x must be a CUDA tensor");
    TORCH_CHECK(x.dim() == 4, "Input must be 4D (N, C, H, W)");

    auto sizes = x.sizes();
    int N = sizes[0];
    int C = sizes[1];
    int H = sizes[2];
    int W = sizes[3];

    auto y = torch::empty_like(x);
    int blocks = N * C;
    int threads = 256;

    instance_norm_kernel_stride<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N, C, H, W, static_cast<float>(eps)
    );

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "InstanceNorm with stride loops (CUDA)");
}
