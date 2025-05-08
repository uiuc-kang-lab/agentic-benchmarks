#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Training kernel specialized to avoid divergent branches by separating training/inference paths
// and using warp shuffle reduction for uniform control flow
__global__ void batch_norm_kernel_train(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    // Each block handles one channel
    const int c = blockIdx.x;
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;

    float sum = 0.0f, sum_sq = 0.0f;
    // Accumulate sums
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }

    // Perform warp-level reduction using shuffle intrinsics
    const unsigned int full_mask = 0xffffffff;
    // Use built-in warpSize which is 32 on NVIDIA GPUs
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(full_mask, sum, offset);
        sum_sq += __shfl_down_sync(full_mask, sum_sq, offset);
    }

    int lane = tid & 31;               // lane index within the warp
    int warpId = tid >> 5;             // warp id in the block

    // Each warp writes its partial sum into shared memory
    __shared__ float warp_sum[32];     // allocate for up to 32 warps
    __shared__ float warp_sum_sq[32];
    if (lane == 0) {
        warp_sum[warpId] = sum;
        warp_sum_sq[warpId] = sum_sq;
    }
    __syncthreads();

    // First warp reduces the partial sums
    float total_sum = 0.0f, total_sum_sq = 0.0f;
    int numWarps = blockDim.x / 32;
    if (tid < numWarps) {
        total_sum = warp_sum[tid];
        total_sum_sq = warp_sum_sq[tid];
        for (int offset = numWarps / 2; offset > 0; offset /= 2) {
            total_sum += __shfl_down_sync(full_mask, total_sum, offset);
            total_sum_sq += __shfl_down_sync(full_mask, total_sum_sq, offset);
        }
        if (tid == 0) {
            float mean = total_sum / num_elements;
            float var = (total_sum_sq / num_elements) - (mean * mean);
            // Update running statistics
            float old_mean = running_mean[c];
            float old_var = running_var[c];
            running_mean[c] = (1.0f - momentum) * old_mean + momentum * mean;
            running_var[c] = (1.0f - momentum) * old_var + momentum * var;
            // Store computed mean and variance in shared memory for broadcasting
            warp_sum[0] = mean;
            warp_sum_sq[0] = var;
        }
    }
    __syncthreads();
    float mean = warp_sum[0];
    float var = warp_sum_sq[0];
    float inv_std = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    // Normalize and write output in a uniform loop
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w_idx = rem % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float in_val = input[idx];
        output[idx] = (in_val - mean) * inv_std * w_val + b_val;
    }
}

// Inference kernel avoids any reduction and divergent branches by using precomputed running statistics
__global__ void batch_norm_kernel_infer(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    float eps,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    const int c = blockIdx.x;
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;

    float mean = running_mean[c];
    float var = running_var[c];
    float inv_std = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w_idx = rem % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float in_val = input[idx];
        output[idx] = (in_val - mean) * inv_std * w_val + b_val;
    }
}

// Forward function chooses the appropriate kernel based on the training flag
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

    if (training) {
        batch_norm_kernel_train<<<C, threads>>>(
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
        batch_norm_kernel_infer<<<C, threads>>>(
            input.data_ptr<float>(),
            weight.data_ptr<float>(),
            bias.data_ptr<float>(),
            running_mean.data_ptr<float>(),
            running_var.data_ptr<float>(),
            eps,
            output.data_ptr<float>(),
            N, C, H, W
        );
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward_cuda, "BatchNorm forward (CUDA) with minimized warp divergence");
}
