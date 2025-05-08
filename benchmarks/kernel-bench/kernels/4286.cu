#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Kernel using warp-level primitives for efficient reduction
__global__ void batch_norm_kernel_warp(
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
    const int blockSize = blockDim.x;
    const int stride = blockSize;

    // Shared memory layout:
    // [0, numWarps-1]: partial sums
    // [numWarps, 2*numWarps-1]: partial sums of squares
    // Later, shared_data[0] and shared_data[1] will hold the final mean and var
    extern __shared__ float shared_data[];
    const int warpSize = 32;
    int numWarps = blockSize / warpSize;
    int warp_id = tid / warpSize;
    int lane = tid % warpSize;

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Loop over the assigned elements to accumulate sum and sum of squares
    for (int i = tid; i < num_elements; i += blockSize) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }

    // Intra-warp reduction using __shfl_down_sync
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
        sum_sq += __shfl_down_sync(mask, sum_sq, offset);
    }

    // Each warp writes its result to shared memory (only lane 0 writes)
    if (lane == 0) {
        shared_data[warp_id] = sum;
        shared_data[warp_id + numWarps] = sum_sq;
    }
    __syncthreads();

    float mean, var;
    // Final reduction across warps is done by thread 0
    if (tid == 0) {
        float final_sum = 0.0f;
        float final_sum_sq = 0.0f;
        for (int i = 0; i < numWarps; i++) {
            final_sum += shared_data[i];
            final_sum_sq += shared_data[i + numWarps];
        }
        mean = final_sum / num_elements;
        var = (final_sum_sq / num_elements) - (mean * mean);
        if (training) {
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
        // Store the computed mean and var for other threads
        shared_data[0] = mean;
        shared_data[1] = var;
    }
    __syncthreads();

    // All threads load the mean and variance
    mean = shared_data[0];
    var = shared_data[1];
    float inv_std = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    // Phase 2: Normalize the input and write to output
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

    // Input validation
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
    int warpSize = 32;
    int numWarps = threads / warpSize;
    // Allocate shared memory: 2 * numWarps floats
    const size_t shared_mem = 2 * numWarps * sizeof(float);

    // Launch one block per channel
    batch_norm_kernel_warp<<<C, threads, shared_mem>>>(
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
    m.def("forward", &forward_cuda, "BatchNorm forward with warp-level reduction (CUDA)");
}
