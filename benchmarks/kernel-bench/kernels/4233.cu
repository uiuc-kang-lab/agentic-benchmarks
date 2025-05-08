#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Optimized CUDA kernel for BatchNorm using warp-level primitives for reduction
__global__ void optimized_batch_norm_kernel(
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

    const int c = blockIdx.x;  // each block handles one channel
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;
    const int blockSize = blockDim.x;
    
    // We'll use warp-level reduction. Assume warp size of 32.
    const int warpSize = 32;
    const int num_warps = blockSize / warpSize;  // assuming blockSize is a multiple of 32

    // Allocate dynamic shared memory:
    // First num_warps floats for warp sums, next num_warps for warp sum of squares,
    // and then 2 floats to store the final mean and variance
    extern __shared__ float shared_mem[];
    float* warp_sums = shared_mem;                   // size: num_warps
    float* warp_sums_sq = warp_sums + num_warps;       // size: num_warps
    float* mean_var = warp_sums_sq + num_warps;        // size: 2 (mean, var)

    float mean, var;

    if (training) {
        // Phase 1: Compute local partial sums
        float local_sum = 0.0f;
        float local_sum_sq = 0.0f;
        for (int i = tid; i < num_elements; i += blockSize) {
            int n = i / (H * W);
            int hw = i % (H * W);
            int h = hw / W;
            int w_idx = hw % W;
            int idx = ((n * C + c) * H + h) * W + w_idx;
            float val = input[idx];
            local_sum += val;
            local_sum_sq += val * val;
        }

        // Warp-level reduction within each warp
        unsigned int mask = 0xffffffff;
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            local_sum += __shfl_down_sync(mask, local_sum, offset);
            local_sum_sq += __shfl_down_sync(mask, local_sum_sq, offset);
        }

        // Each warp writes its reduced result to shared memory
        int lane = tid & (warpSize - 1);
        int warp_id = tid / warpSize;
        if (lane == 0) {
            warp_sums[warp_id] = local_sum;
            warp_sums_sq[warp_id] = local_sum_sq;
        }
        __syncthreads();

        // Final reduction across warps. Use first num_warps threads (they are in one warp if num_warps <= warpSize).
        if (tid < num_warps) {
            float final_sum = warp_sums[tid];
            float final_sum_sq = warp_sums_sq[tid];
            // Since num_warps is small, use warp-level reduction again
            for (int offset = num_warps >> 1; offset > 0; offset >>= 1) {
                final_sum += __shfl_down_sync(0xffffffff, final_sum, offset);
                final_sum_sq += __shfl_down_sync(0xffffffff, final_sum_sq, offset);
            }
            if (tid == 0) {
                mean = final_sum / num_elements;
                var = (final_sum_sq / num_elements) - (mean * mean);
                // Update running statistics
                running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
                running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
                // Store computed mean and variance for phase 2
                mean_var[0] = mean;
                mean_var[1] = var;
            }
        }
        __syncthreads();
        mean = mean_var[0];
        var = mean_var[1];
    } else {
        // Use stored running statistics in inference mode
        mean = running_mean[c];
        var = running_var[c];
    }

    // Phase 2: Normalize and write output
    float inv_std = rsqrtf(var + eps);
    float gamma = weight[c];
    float beta = bias[c];
    
    for (int i = tid; i < num_elements; i += blockSize) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w_idx = hw % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float val = input[idx];
        output[idx] = (val - mean) * inv_std * gamma + beta;
    }
}

// Host function to launch the optimized kernel
torch::Tensor forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    // Input checks
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
    // Shared memory size: 2 * (threads/32) for warp arrays + 2 floats for mean and var
    const int num_warps = threads / 32;
    const size_t shared_mem = (2 * num_warps + 2) * sizeof(float);

    optimized_batch_norm_kernel<<<C, threads, shared_mem>>>(
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
    m.def("forward", &forward_cuda, "BatchNorm forward (CUDA) with warp-level optimizations");
}
