#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define WARP_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ void computeWarpSums(const float4* input4,
                               int c, int N, int C, int H, int W,
                               int tid, int stride,
                               float &sum, float &sumSq) {
    sum = 0.0f;
    sumSq = 0.0f;
    
    const int numElements4 = (N * H * W) / 4;
    const int remainder = (N * H * W) % 4;
    
    // Vector loads for better memory bandwidth
    for (int i = tid; i < numElements4; i += stride) {
        const int base_idx = ((i * 4) / (H * W));
        const int base_hw = ((i * 4) % (H * W));
        const int base_h = base_hw / W;
        const int base_w = base_hw % W;
        const int base = ((base_idx * C + c) * H + base_h) * W + base_w;
        
        float4 vals = input4[base/4];
        sum += vals.x + vals.y + vals.z + vals.w;
        sumSq += vals.x * vals.x + vals.y * vals.y + 
                 vals.z * vals.z + vals.w * vals.w;
    }
    
    // Handle remainder elements
    const float* input = reinterpret_cast<const float*>(input4);
    for (int i = numElements4 * 4 + tid; i < numElements4 * 4 + remainder; i += stride) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int h = hw / W;
        const int w = hw % W;
        const int idx = ((n * C + c) * H + h) * W + w;
        const float val = input[idx];
        sum += val;
        sumSq += val * val;
    }
}

__global__ void warp_atomic_batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    bool training,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N, int C, int H, int W) {
    
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int stride = blockDim.x;
    const int num_elements = N * H * W;
    
    __shared__ float warp_sums[32];    // For mean
    __shared__ float warp_sumsq[32];   // For variance
    __shared__ float mean_var[2];      // Final statistics
    
    float sum = 0.0f, sumSq = 0.0f;
    
    // Compute partial sums using vectorized loads where possible
    computeWarpSums(reinterpret_cast<const float4*>(input),
                   c, N, C, H, W, tid, stride, sum, sumSq);
    
    // Warp-level reduction
    float warp_sum = warpReduceSum(sum);
    float warp_sumsq = warpReduceSum(sumSq);
    
    if (lane == 0) {
        warp_sums[wid] = warp_sum;
        warp_sumsq[wid] = warp_sumsq;
    }
    __syncthreads();
    
    // Final reduction and statistics computation
    if (tid == 0) {
        float final_sum = 0.0f;
        float final_sumsq = 0.0f;
        
        #pragma unroll
        for (int i = 0; i < warps_per_block; ++i) {
            final_sum += warp_sums[i];
            final_sumsq += warp_sumsq[i];
        }
        
        float mean = final_sum / num_elements;
        float var = (final_sumsq / num_elements) - (mean * mean);
        
        if (training) {
            // Atomic update of running statistics - only one thread per channel
            atomicAdd(&running_mean[c], momentum * (mean - running_mean[c]));
            atomicAdd(&running_var[c], momentum * (var - running_var[c]));
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        
        mean_var[0] = mean;
        mean_var[1] = var;
    }
    __syncthreads();
    
    const float mean = mean_var[0];
    const float var = mean_var[1];
    const float inv_std = rsqrtf(var + eps);
    const float w = weight[c];
    const float b = bias[c];
    
    // Normalize output using vectorized stores where possible
    for (int i = tid; i < num_elements; i += stride) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int h = hw / W;
        const int w_idx = hw % W;
        const int idx = ((n * C + c) * H + h) * W + w_idx;
        const float val = input[idx];
        output[idx] = (val - mean) * inv_std * w + b;
    }
}

torch::Tensor forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {
    
    CHECK_CUDA(input);
    CHECK_CUDA(weight);
    CHECK_CUDA(bias);
    CHECK_CUDA(running_mean);
    CHECK_CUDA(running_var);
    
    CHECK_CONTIGUOUS(input);
    CHECK_CONTIGUOUS(weight);
    CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(running_mean);
    CHECK_CONTIGUOUS(running_var);
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    
    warp_atomic_batch_norm_kernel<<<C, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        training,
        momentum,
        eps,
        output.data_ptr<float>(),
        N, C, H, W
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "BatchNorm forward (CUDA)");
}