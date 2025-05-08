#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ void warpReduceSumDual(float& val1, float& val2) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val1 += __shfl_down_sync(0xffffffff, val1, offset);
        val2 += __shfl_down_sync(0xffffffff, val2, offset);
    }
}

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
    const int HW = H * W;
    const int instance_idx = blockIdx.x;
    if (instance_idx >= N * C) return;

    const int n = instance_idx / C;
    const int c = instance_idx % C;
    
    // Shared memory for partial sums
    __shared__ float s_partial_sums[32][2];  // [warp_count][sum, sum_sq]
    
    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;

    // Process data in chunks of float4
    const int vec_elements = HW / 4;
    const int vec_stride = blockDim.x;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Vector load phase
    for (int i = threadIdx.x; i < vec_elements; i += vec_stride) {
        float4 data = reinterpret_cast<const float4*>(x_instance)[i];
        sum += data.x + data.y + data.z + data.w;
        sum_sq += data.x * data.x + data.y * data.y + 
                  data.z * data.z + data.w * data.w;
    }

    // Handle remaining elements
    const int rem_start = vec_elements * 4;
    for (int i = rem_start + threadIdx.x; i < HW; i += blockDim.x) {
        float val = x_instance[i];
        sum += val;
        sum_sq += val * val;
    }

    // Warp-level reduction
    const int warp_id = threadIdx.x / warpSize;
    const int lane_id = threadIdx.x % warpSize;
    
    warpReduceSumDual(sum, sum_sq);

    // Store warp results in shared memory
    if (lane_id == 0) {
        s_partial_sums[warp_id][0] = sum;
        s_partial_sums[warp_id][1] = sum_sq;
    }
    __syncthreads();

    // Final reduction by first warp
    if (warp_id == 0 && lane_id < (blockDim.x + warpSize - 1) / warpSize) {
        sum = s_partial_sums[lane_id][0];
        sum_sq = s_partial_sums[lane_id][1];
        
        warpReduceSumDual(sum, sum_sq);
    }

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum / HW;
        float var = (sum_sq / HW) - (mean * mean);
        var = max(var, 0.0f);  // Ensure non-negative variance
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    const float mean = s_mean;
    const float inv_std = s_inv_std;
    const float w = weight ? weight[c] : 1.0f;
    const float b = bias ? bias[c] : 0.0f;

    // Normalize using vectorized operations
    for (int i = threadIdx.x; i < vec_elements; i += vec_stride) {
        float4 data = reinterpret_cast<const float4*>(x_instance)[i];
        float4 result;
        result.x = ((data.x - mean) * inv_std * w) + b;
        result.y = ((data.y - mean) * inv_std * w) + b;
        result.z = ((data.z - mean) * inv_std * w) + b;
        result.w = ((data.w - mean) * inv_std * w) + b;
        reinterpret_cast<float4*>(y_instance)[i] = result;
    }

    // Handle remaining elements
    for (int i = rem_start + threadIdx.x; i < HW; i += blockDim.x) {
        float val = x_instance[i];
        y_instance[i] = ((val - mean) * inv_std * w) + b;
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
    TORCH_CHECK(sizes.size() == 4, "Input must be 4D");
    
    const int N = sizes[0], C = sizes[1], H = sizes[2], W = sizes[3];
    auto y = torch::empty_like(x);
    
    const int threads = 256;
    const int blocks = N * C;
    
    instance_norm_kernel_optimized<<<blocks, threads>>>(
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