#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define WARP_SIZE 32

__global__ void batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    bool training,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N,
    int C,
    int H,
    int W) {
    
    const int c = blockIdx.x;
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;
    const int lane_id = tid % WARP_SIZE;
    const int warp_id = tid / WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    
    extern __shared__ float smem[];
    float* warp_sum = smem;
    float* warp_sum_sq = &smem[warps_per_block];
    
    float mean, var;
    float sum = 0.0f, sum_sq = 0.0f;
    
    // Pre-compute array indices for better memory coalescing
    const int base_idx = c * H * W;
    
    // Phase 1: Compute sum and sum of squares (warp-aligned)
    #pragma unroll 4
    for (int i = tid; i < num_elements; i += blockDim.x) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int idx = (n * C * H * W) + base_idx + hw;
        const float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }
    
    // Warp-level reduction using shuffle
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
    
    // First thread in each warp writes to shared memory
    if (lane_id == 0) {
        warp_sum[warp_id] = sum;
        warp_sum_sq[warp_id] = sum_sq;
    }
    __syncthreads();
    
    // Final reduction across warps
    if (warp_id == 0 && lane_id < warps_per_block) {
        float block_sum = warp_sum[lane_id];
        float block_sum_sq = warp_sum_sq[lane_id];
        
        // Warp-level reduction for final results
        #pragma unroll
        for (int offset = warps_per_block/2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
            block_sum_sq += __shfl_down_sync(0xffffffff, block_sum_sq, offset);
        }
        
        if (lane_id == 0) {
            mean = block_sum / num_elements;
            var = (block_sum_sq / num_elements) - (mean * mean);
            
            if (training) {
                running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
                running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
            } else {
                mean = running_mean[c];
                var = running_var[c];
            }
            
            // Store in shared mem for all threads to use
            warp_sum[0] = mean;
            warp_sum[1] = var;
        }
    }
    __syncthreads();
    
    // Load final mean and var
    mean = warp_sum[0];
    var = warp_sum[1];
    
    // Phase 2: Normalize and write output (coalesced memory access)
    const float inv_std = rsqrtf(var + eps);
    const float w = weight[c];
    const float b = bias[c];
    
    #pragma unroll 4
    for (int i = tid; i < num_elements; i += blockDim.x) {
        const int n = i / (H * W);
        const int hw = i % (H * W);
        const int idx = (n * C * H * W) + base_idx + hw;
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
    const int warps_per_block = threads / WARP_SIZE;
    const size_t shared_mem = 2 * warps_per_block * sizeof(float);
    
    batch_norm_kernel<<<C, threads, shared_mem>>>(
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