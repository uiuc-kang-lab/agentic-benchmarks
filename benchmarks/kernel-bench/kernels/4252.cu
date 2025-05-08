#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define WARP_SIZE 32

__inline__ __device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<bool Training>
__global__ void divergence_free_batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N, int C, int H, int W) {
    
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int wid = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    const int num_elements = N * H * W;
    
    __shared__ float warp_means[32];
    __shared__ float warp_vars[32];
    
    float mean, inv_std;
    
    if (Training) {
        // Compute sums using grid-stride loop with uniform control flow
        float sum = 0.0f;
        float sum_sq = 0.0f;
        
        #pragma unroll 4
        for (int idx = tid; idx < num_elements; idx += blockDim.x) {
            const int n = idx / (H * W);
            const int hw = idx % (H * W);
            const float val = input[((n * C + c) * H + (hw/W)) * W + (hw%W)];
            sum += val;
            sum_sq += val * val;
        }
        
        // Warp reduction - no divergent branches
        sum = warpReduceSum(sum);
        sum_sq = warpReduceSum(sum_sq);
        
        // Store warp results
        if (lane == 0) {
            warp_means[wid] = sum;
            warp_vars[wid] = sum_sq;
        }
        __syncthreads();
        
        // Final reduction by first warp only
        if (wid == 0) {
            sum = (lane < warps_per_block) ? warp_means[lane] : 0.0f;
            sum_sq = (lane < warps_per_block) ? warp_vars[lane] : 0.0f;
            
            sum = warpReduceSum(sum);
            sum_sq = warpReduceSum(sum_sq);
            
            if (lane == 0) {
                mean = sum / num_elements;
                float var = (sum_sq / num_elements) - (mean * mean);
                inv_std = rsqrtf(var + eps);
                
                // Update running stats
                running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
                running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
                
                // Store in shared memory for all threads
                warp_means[0] = mean;
                warp_vars[0] = inv_std;
            }
        }
        __syncthreads();
        
        mean = warp_means[0];
        inv_std = warp_vars[0];
    } else {
        // Inference path - uniform control flow
        mean = running_mean[c];
        inv_std = rsqrtf(running_var[c] + eps);
    }
    
    const float w = weight[c];
    const float b = bias[c];
    
    // Normalize with vectorized memory access
    #pragma unroll 4
    for (int idx = tid; idx < num_elements; idx += blockDim.x) {
        const int n = idx / (H * W);
        const int hw = idx % (H * W);
        const int out_idx = ((n * C + c) * H + (hw/W)) * W + (hw%W);
        const float val = input[out_idx];
        output[out_idx] = (val - mean) * inv_std * w + b;
    }
}

torch::Tensor divergence_free_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {
    
    CHECK_CUDA(input); CHECK_CUDA(weight); CHECK_CUDA(bias);
    CHECK_CUDA(running_mean); CHECK_CUDA(running_var);
    CHECK_CONTIGUOUS(input); CHECK_CONTIGUOUS(weight); CHECK_CONTIGUOUS(bias);
    CHECK_CONTIGUOUS(running_mean); CHECK_CONTIGUOUS(running_var);
    
    const int N = input.size(0);
    const int C = input.size(1);
    const int H = input.size(2);
    const int W = input.size(3);
    
    auto output = torch::empty_like(input);
    
    const int threads = 256;
    
    if (training) {
        divergence_free_batch_norm_kernel<true><<<C, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            momentum,
            eps,
            output.data_ptr<float>(),
            N, C, H, W
        );
    } else {
        divergence_free_batch_norm_kernel<false><<<C, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            momentum,
            eps,
            output.data_ptr<float>(),
            N, C, H, W
        );
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &divergence_free_forward_cuda, "Divergence-free BatchNorm forward (CUDA)");
}