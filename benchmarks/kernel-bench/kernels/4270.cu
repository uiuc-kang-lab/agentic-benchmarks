#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define WARP_SIZE 32
#define BLOCK_SIZE 512

__device__ __forceinline__ float2 warpReduceSumF2(float2 val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val.x += __shfl_down_sync(0xffffffff, val.x, offset);
        val.y += __shfl_down_sync(0xffffffff, val.y, offset);
    }
    return val;
}

__global__ void optimized_warp_batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    float* __restrict__ output,
    const int N, const int C, const int H, const int W,
    const bool training, const float momentum, const float eps) {
    
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warps_per_block = BLOCK_SIZE / WARP_SIZE;
    const int num_elements = N * H * W;
    
    __shared__ float s_mean_per_warp[warps_per_block];
    __shared__ float s_var_per_warp[warps_per_block];
    
    float2 local_sum = make_float2(0.0f, 0.0f);
    
    const int vec_size = 4;
    const int vec_elements = num_elements / vec_size;
    const float4* input4 = reinterpret_cast<const float4*>(input + c * num_elements);
    
    #pragma unroll 4
    for (int i = tid; i < vec_elements; i += BLOCK_SIZE) {
        float4 values = input4[i];
        local_sum.x += values.x + values.y + values.z + values.w;
        local_sum.y += values.x * values.x + values.y * values.y + 
                      values.z * values.z + values.w * values.w;
    }
    
    const int remaining_start = vec_elements * vec_size;
    #pragma unroll
    for (int i = remaining_start + tid; i < num_elements; i += BLOCK_SIZE) {
        float val = input[c * num_elements + i];
        local_sum.x += val;
        local_sum.y += val * val;
    }
    
    float2 warp_sum = warpReduceSumF2(local_sum);
    
    if (lane == 0) {
        s_mean_per_warp[wid] = warp_sum.x;
        s_var_per_warp[wid] = warp_sum.y;
    }
    __syncthreads();
    
    if (wid == 0) {
        float2 block_sum = make_float2(0.0f, 0.0f);
        if (lane < warps_per_block) {
            block_sum.x = s_mean_per_warp[lane];
            block_sum.y = s_var_per_warp[lane];
        }
        
        block_sum = warpReduceSumF2(block_sum);
        
        if (lane == 0) {
            float mean = block_sum.x / num_elements;
            float variance = block_sum.y / num_elements - mean * mean;
            
            if (training) {
                running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
                running_var[c] = (1.0f - momentum) * running_var[c] + momentum * variance;
            } else {
                mean = running_mean[c];
                variance = running_var[c];
            }
            
            s_mean_per_warp[0] = mean;
            s_var_per_warp[0] = variance;
        }
    }
    __syncthreads();
    
    const float mean = s_mean_per_warp[0];
    const float variance = s_var_per_warp[0];
    const float inv_std = rsqrtf(variance + eps);
    const float w = weight[c];
    const float b = bias[c];
    
    float4* output4 = reinterpret_cast<float4*>(output + c * num_elements);
    #pragma unroll 4
    for (int i = tid; i < vec_elements; i += BLOCK_SIZE) {
        float4 values = input4[i];
        values.x = (values.x - mean) * inv_std * w + b;
        values.y = (values.y - mean) * inv_std * w + b;
        values.z = (values.z - mean) * inv_std * w + b;
        values.w = (values.w - mean) * inv_std * w + b;
        output4[i] = values;
    }
    
    #pragma unroll
    for (int i = remaining_start + tid; i < num_elements; i += BLOCK_SIZE) {
        float val = input[c * num_elements + i];
        output[c * num_elements + i] = (val - mean) * inv_std * w + b;
    }
}

torch::Tensor optimized_forward_cuda(
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
    
    optimized_warp_batch_norm_kernel<<<C, BLOCK_SIZE>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W,
        training,
        momentum,
        eps
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimized_forward_cuda, "Optimized Warp BatchNorm forward (CUDA)");
}