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

__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32];
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);
    if (lane == 0) shared[wid] = val;
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
    if (wid == 0) val = warpReduceSum(val);
    return val;
}

__global__ void instance_norm_kernel(
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
    const int n = instance_idx / C;
    const int c = instance_idx % C;
    
    if (instance_idx >= N * C) return;

    const float* x_instance = x + (n * C + c) * HW;
    float* y_instance = y + (n * C + c) * HW;

    // Use float4 for better memory coalescing
    const int vec_elements = HW / 4;
    const int vec_stride = blockDim.x;
    
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // Process chunks of 4 elements using float4
    const float4* x_vec = reinterpret_cast<const float4*>(x_instance);
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_elements; i += vec_stride) {
        float4 data = x_vec[i];
        local_sum += data.x + data.y + data.z + data.w;
        local_sq_sum += data.x * data.x + data.y * data.y + 
                       data.z * data.z + data.w * data.w;
    }

    // Handle remaining elements
    const int rem_start = vec_elements * 4;
    for (int i = rem_start + threadIdx.x; i < HW; i += vec_stride) {
        float val = x_instance[i];
        local_sum += val;
        local_sq_sum += val * val;
    }

    // Reduce sums within block
    float sum = blockReduceSum(local_sum);
    float sum_sq = blockReduceSum(local_sq_sum);

    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        float mean = sum / HW;
        float var = (sum_sq / HW) - (mean * mean);
        var = max(var, 0.0f);
        s_mean = mean;
        s_inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    const float mean = s_mean;
    const float inv_std = s_inv_std;
    const float scale = weight ? weight[c] : 1.0f;
    const float shift = bias ? bias[c] : 0.0f;

    // Normalize using vectorized operations
    float4* y_vec = reinterpret_cast<float4*>(y_instance);
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_elements; i += vec_stride) {
        float4 data = x_vec[i];
        float4 result;
        result.x = ((data.x - mean) * inv_std * scale) + shift;
        result.y = ((data.y - mean) * inv_std * scale) + shift;
        result.z = ((data.z - mean) * inv_std * scale) + shift;
        result.w = ((data.w - mean) * inv_std * scale) + shift;
        y_vec[i] = result;
    }

    // Handle remaining elements
    for (int i = rem_start + threadIdx.x; i < HW; i += vec_stride) {
        float val = x_instance[i];
        y_instance[i] = ((val - mean) * inv_std * scale) + shift;
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
    const int HW = H * W;
    auto y = torch::empty_like(x);
    
    // Dynamic block size selection based on feature map size
    int threads;
    if (HW < 1024) {
        threads = 128;
    } else if (HW < 4096) {
        threads = 256;
    } else {
        threads = 512;
    }
    
    const dim3 blocks(N * C);
    
    instance_norm_kernel<<<blocks, threads>>>(
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