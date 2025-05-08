#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Define tunable block size. Experiment with values: 32, 64, 128, 256, 512.
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// Warp-level reduction for sum using __shfl_down_sync
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory, with tunable block size
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // one element per warp (max 32 warps)
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Only threads in the first warp participate in final reduction
    int numWarps = (BLOCK_SIZE + warpSize - 1) / warpSize;
    val = (threadIdx.x < numWarps) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

// CUDA kernel for instance normalization with tunable block size
__global__ void instance_norm_kernel_tunable(
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
    // Each block processes one (N, C) instance
    int instance = blockIdx.x;
    if (instance >= N * C) return;

    int n = instance / C;
    int c = instance % C;
    int HW = H * W;

    // Pointers to current instance data
    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;

    // Use vectorized loads/stores with float4 (ensuring alignment).
    int vec_elements = HW / 4;  // number of float4 elements
    int rem = HW % 4;           // remaining elements

    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float4* x_vec = reinterpret_cast<const float4*>(x_instance);

    // First pass: compute partial sums using tunable block size (BLOCK_SIZE)
    for (int i = threadIdx.x; i < vec_elements; i += BLOCK_SIZE) {
        float4 data = x_vec[i];
        sum += data.x + data.y + data.z + data.w;
        sum_sq += data.x * data.x + data.y * data.y + data.z * data.z + data.w * data.w;
    }

    int base = vec_elements * 4;
    for (int i = threadIdx.x; i < rem; i += BLOCK_SIZE) {
        float data = x_instance[base + i];
        sum += data;
        sum_sq += data * data;
    }

    // Reduce sums over the block
    sum = blockReduceSum(sum);
    sum_sq = blockReduceSum(sum_sq);

    __shared__ float s_mean;
    __shared__ float s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum / HW;
        float var = (sum_sq / HW) - (mean * mean);
        var = (var < 0.0f) ? 0.0f : var;
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    float mean = s_mean;
    float inv_std = s_inv_std;
    float scale = (weight != nullptr) ? weight[c] : 1.0f;
    float shift = (bias != nullptr) ? bias[c] : 0.0f;

    // Second pass: normalize data
    float4* y_vec = reinterpret_cast<float4*>(y_instance);
    for (int i = threadIdx.x; i < vec_elements; i += BLOCK_SIZE) {
        float4 data = x_vec[i];
        float4 res;
        res.x = ((data.x - mean) * inv_std * scale) + shift;
        res.y = ((data.y - mean) * inv_std * scale) + shift;
        res.z = ((data.z - mean) * inv_std * scale) + shift;
        res.w = ((data.w - mean) * inv_std * scale) + shift;
        y_vec[i] = res;
    }
    for (int i = threadIdx.x; i < rem; i += BLOCK_SIZE) {
        int idx = base + i;
        float data = x_instance[idx];
        y_instance[idx] = ((data - mean) * inv_std * scale) + shift;
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
    int total_instances = N * C;
    dim3 blocks(total_instances);
    dim3 threads(BLOCK_SIZE);

    instance_norm_kernel_tunable<<<blocks, threads>>>(
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
    m.def("forward", &forward, "Instance Normalization forward (CUDA) with tunable block size");
}
