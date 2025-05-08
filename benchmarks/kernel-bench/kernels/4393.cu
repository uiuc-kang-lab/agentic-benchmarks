#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Simple warp reduce sum for float
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

// Block reduce sum using warp reduces
__inline__ __device__ float blockReduceSum(float val) {
    static __shared__ float shared[32]; // 1 warp per 32 threads
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Only the first warp is active here.
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// CUDA kernel for instance normalization
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
    int instance_id = blockIdx.x; // each block handles one (N, C) pair
    int n = instance_id / C;
    int c = instance_id % C;

    int HW = H * W;
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr       = y + (n * C + c) * HW;

    extern __shared__ float tile[];
    // Load input data into shared memory tile
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        tile[i] = x_ptr[i];
    }
    __syncthreads();

    float sum_val = 0.0f, sum_sq_val = 0.0f;
    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float v = tile[i];
        sum_val += v;
        sum_sq_val += v * v;
    }

    sum_val = blockReduceSum(sum_val);
    sum_sq_val = blockReduceSum(sum_sq_val);

    __shared__ float sharedMean;
    __shared__ float sharedVar;

    if (threadIdx.x == 0) {
        float mean = sum_val / (float)HW;
        float var = sum_sq_val / (float)HW - mean * mean;
        var = (var < 0.f) ? 0.f : var;
        sharedMean = mean;
        sharedVar = var;
    }
    __syncthreads();

    float mean = sharedMean;
    float var = sharedVar;
    float invert_std = 1.0f / sqrtf(var + eps);

    for (int i = threadIdx.x; i < HW; i += blockDim.x) {
        float v = x_ptr[i];
        float norm_val = (v - mean) * invert_std; // Pre-compute for memory efficiency
        norm_val = norm_val * (weight ? weight[c] : 1.0f) + (bias ? bias[c] : 0.0f);
        y_ptr[i] = norm_val;
    }
}

// Forward function called from Python
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
    instance_norm_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        (weight.defined() && weight.numel() > 0) ? weight.data_ptr<float>() : nullptr,
        (bias.defined() && bias.numel() > 0) ? bias.data_ptr<float>() : nullptr,
        N,
        C,
        H,
        W,
        static_cast<float>(eps)
    );

    return y;
}

// Pybind11 module definition
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Instance Normalization forward (CUDA)");
}
