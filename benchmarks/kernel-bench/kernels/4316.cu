#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Optimized BatchNorm kernel using warp-level primitives to reduce unnecessary synchronizations.
__global__ void batch_norm_kernel_optimized(
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
    
    // Each block is responsible for one channel
    const int c = blockIdx.x;
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;

    float mean, var;

    if (training) {
        // Phase 1: Compute partial sums and squared sums
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;
        for (int i = tid; i < num_elements; i += blockSize) {
            int n = i / (H * W);
            int rem = i % (H * W);
            int h = rem / W;
            int w = rem % W;
            int idx = ((n * C + c) * H + h) * W + w;
            float val = input[idx];
            local_sum += val;
            local_sum_sq += val * val;
        }

        // Warp-level reduction using shuffle intrinsics
        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
            local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
        }

        // Number of warps in this block
        int numWarps = (blockSize + warpSize - 1) / warpSize;
        int warpId = tid / warpSize;
        int lane = tid % warpSize;

        // Allocate shared memory for inter-warp reduction
        // Layout: first numWarps floats for sum, next numWarps floats for sum_sq
        extern __shared__ float shared_data[];
        if (lane == 0) {
            shared_data[warpId] = local_sum;
            shared_data[numWarps + warpId] = local_sum_sq;
        }

        // Synchronize to ensure all warp results are in shared memory
        __syncthreads();

        float block_sum = 0.0f;
        float block_sum_sq = 0.0f;
        // Let thread 0 perform the final reduction over warp sums
        if (tid == 0) {
            for (int i = 0; i < numWarps; i++) {
                block_sum += shared_data[i];
                block_sum_sq += shared_data[numWarps + i];
            }
            mean = block_sum / num_elements;
            var = (block_sum_sq / num_elements) - (mean * mean);
            
            // Update running statistics
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;
            
            // Store computed mean and variance in shared memory for broadcast
            shared_data[0] = mean;
            shared_data[1] = var;
        }
        // One synchronization to ensure mean and var have been broadcast
        __syncthreads();
        mean = shared_data[0];
        var = shared_data[1];
    } else {
        mean = running_mean[c];
        var = running_var[c];
    }

    // Phase 2: Normalize and write output
    float inv_std = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    for (int i = tid; i < num_elements; i += blockSize) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w_idx = rem % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float val = input[idx];
        output[idx] = (val - mean) * inv_std * w_val + b_val;
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
    // Calculate required shared memory: 2 arrays of size equal to number of warps
    int numWarps = (threads + warpSize - 1) / warpSize;
    size_t shared_mem = 2 * numWarps * sizeof(float);
    
    batch_norm_kernel_optimized<<<C, threads, shared_mem>>>(
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
    m.def("forward", &forward_cuda, "BatchNorm forward optimized (CUDA)");
}
