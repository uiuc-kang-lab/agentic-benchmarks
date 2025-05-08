#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define WARP_SIZE 32

__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void warp_aligned_batch_norm_kernel(
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
    const int stride = blockDim.x;

    extern __shared__ float smem[];
    float* warp_sums = smem;
    float* warp_sq_sums = &smem[warps_per_block];

    float mean, var;
    
    if (training) {
        // First compute warp-level partial sums
        float sum = 0.0f, sum_sq = 0.0f;
        
        // Coalesced memory access pattern
        for (int i = tid; i < num_elements; i += stride) {
            const int idx = ((i / (H * W)) * C + c) * H * W + (i % (H * W));
            const float val = input[idx];
            sum += val;
            sum_sq += val * val;
        }

        // Warp-level reduction
        sum = warp_reduce_sum(sum);
        sum_sq = warp_reduce_sum(sum_sq);

        // First thread in each warp writes to shared memory
        if (lane_id == 0) {
            warp_sums[warp_id] = sum;
            warp_sq_sums[warp_id] = sum_sq;
        }
        __syncthreads();

        // Final reduction across warps (only first warp)
        if (warp_id == 0 && lane_id < warps_per_block) {
            float block_sum = warp_sums[lane_id];
            float block_sum_sq = warp_sq_sums[lane_id];
            
            block_sum = warp_reduce_sum(block_sum);
            block_sum_sq = warp_reduce_sum(block_sum_sq);

            if (lane_id == 0) {
                mean = block_sum / num_elements;
                var = (block_sum_sq / num_elements) - (mean * mean);
                
                running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
                running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
                
                // Store in shared mem for next phase
                warp_sums[0] = mean;
                warp_sums[1] = var;
            }
        }
        __syncthreads();
        
        mean = warp_sums[0];
        var = warp_sums[1];
    } else {
        mean = running_mean[c];
        var = running_var[c];
    }

    // Broadcast constants to all threads
    const float inv_std = rsqrtf(var + eps);
    const float w = weight[c];
    const float b = bias[c];
    
    // Vectorized output computation with coalesced memory access
    #pragma unroll 4
    for (int i = tid; i < num_elements; i += stride) {
        const int idx = ((i / (H * W)) * C + c) * H * W + (i % (H * W));
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
    
    const int threads = 256; // Must be multiple of WARP_SIZE
    const int warps_per_block = threads / WARP_SIZE;
    const size_t shared_mem = 2 * warps_per_block * sizeof(float);
    
    warp_aligned_batch_norm_kernel<<<C, threads, shared_mem>>>(
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