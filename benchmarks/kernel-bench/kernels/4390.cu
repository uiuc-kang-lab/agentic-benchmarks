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
    __shared__ float shared[32]; // one warp per 32 threads
    int lane = threadIdx.x % warpSize;
    int wid  = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    __syncthreads();

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    val = (lane < numWarps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// CUDA kernel for instance normalization with manual loop unrolling
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
    int instance_id = blockIdx.x; // Each block handles one (N, C) pair
    int n = instance_id / C;
    int c = instance_id % C;

    int HW = H * W;
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;

    // Manually unroll loop by factor of 4
    const int unroll = 4;
    int stride = blockDim.x * unroll;
    for (int base = threadIdx.x; base < HW; base += stride) {
        #pragma unroll
        for (int j = 0; j < unroll; j++) {
            int idx = base + j * blockDim.x;
            if (idx < HW) {
                float v = x_ptr[idx];
                sum_val += v;
                sum_sq_val += v * v;
            }
        }
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
    float inv_std = 1.0f / sqrtf(var + eps);

    // Precompute scale and shift outside the loop to avoid per-iteration conditionals
    float scale = (weight != nullptr) ? weight[c] : 1.0f;
    float shift = (bias != nullptr) ? bias[c] : 0.0f;

    // Normalize using manual unrolling
    for (int base = threadIdx.x; base < HW; base += stride) {
        #pragma unroll
        for (int j = 0; j < unroll; j++) {
            int idx = base + j * blockDim.x;
            if (idx < HW) {
                float v = x_ptr[idx];
                float norm_val = (v - mean) * inv_std;
                y_ptr[idx] = norm_val * scale + shift;
            }
        }
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
    m.def("forward", &forward, "Instance Normalization forward (CUDA) with loop unrolling");
}
