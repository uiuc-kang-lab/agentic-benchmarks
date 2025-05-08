#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>

// Warp-level reduction for float
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

// Block-level reduction using a 2D thread indexing scheme
__inline__ __device__ float blockReduceSum(float val) {
    // Flatten the 2D thread index into a 1D index
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    // Intra-warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    // Allocate shared memory to hold the sum from each warp
    __shared__ float shared[32]; // Sufficient for up to 32 warps
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    // Use the first warp to reduce the warp-level sums
    if (tid == 0) {
        int num_warps = (blockDim.x * blockDim.y + warpSize - 1) / warpSize;
        float total = 0.0f;
        for (int i = 0; i < num_warps; i++) {
            total += shared[i];
        }
        shared[0] = total;
    }
    __syncthreads();
    return shared[0];
}

// CUDA kernel for Instance Normalization with 2D thread mapping
__global__ void instance_norm_kernel_2d(
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
    // Each block now corresponds to one (N, C) pair.
    // Use a 2D grid: blockIdx.y = n, blockIdx.x = c
    int n = blockIdx.y;
    int c = blockIdx.x;
    int HW = H * W;

    // Pointer to the start of the instance
    const float* x_ptr = x + (n * C + c) * HW;
    float* y_ptr = y + (n * C + c) * HW;

    // Flatten 2D thread index
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int stride = blockDim.x * blockDim.y;

    // First pass: compute mean and variance (sum and sum of squares)
    float sum_val = 0.0f;
    float sum_sq_val = 0.0f;
    for (int i = tid; i < HW; i += stride) {
        float v = x_ptr[i];
        sum_val += v;
        sum_sq_val += v * v;
    }

    sum_val = blockReduceSum(sum_val);
    sum_sq_val = blockReduceSum(sum_sq_val);

    __shared__ float sharedMean;
    __shared__ float sharedVar;

    if (tid == 0) {
        float mean = sum_val / static_cast<float>(HW);
        float var = sum_sq_val / static_cast<float>(HW) - mean * mean;
        // Clamp variance to avoid negatives due to floating point precision issues
        var = (var < 0.0f) ? 0.0f : var;
        sharedMean = mean;
        sharedVar = var;
    }
    __syncthreads();

    float mean = sharedMean;
    float var = sharedVar;

    // Second pass: normalize and optionally apply scale and bias
    for (int i = tid; i < HW; i += stride) {
        float v = x_ptr[i];
        float norm_val = (v - mean) / sqrtf(var + eps);
        if (weight && bias) {
            norm_val = norm_val * weight[c] + bias[c];
        }
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

    // Launch the kernel with a 2D grid: x-dim for channels, y-dim for batch instances
    // and a 2D block (e.g., 16x16 threads) to map the spatial domain (H, W) efficiently
    dim3 threads(16, 16);
    dim3 blocks(C, N);

    instance_norm_kernel_2d<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Instance Normalization forward (CUDA) with 2D thread mapping");
}
