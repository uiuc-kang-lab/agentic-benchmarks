#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Optimized BatchNorm kernel with manual loop unrolling
__global__ void batch_norm_unroll_kernel(
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
    if (c >= C) return;
    const int num_elements = N * H * W;
    const int tid = threadIdx.x;
    const int blockSize = min(blockDim.x, num_elements);

    // Each thread computes its partial sum and sum of squares
    float my_sum = 0.f;
    float my_sum_sq = 0.f;
    #pragma unroll
    for (int i = tid; i < num_elements; i += blockSize) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        my_sum += val;
        my_sum_sq += val * val;
    }

    // Warp-level reduction using shfl, unrolled for known warp size
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        my_sum += __shfl_down_sync(0xffffffff, my_sum, offset);
        my_sum_sq += __shfl_down_sync(0xffffffff, my_sum_sq, offset);
    }

    // Shared memory to accumulate warp sums
    __shared__ float warp_sum[32];
    __shared__ float warp_sum_sq[32];
    int warpId = tid / warpSize;
    int lane = tid % warpSize;
    if (lane == 0) {
        warp_sum[warpId] = my_sum;
        warp_sum_sq[warpId] = my_sum_sq;
    }
    __syncthreads();

    float mean = 0.f, var = 0.f;
    if (tid == 0) {
        int num_warps = blockSize / warpSize;
        float total_sum = 0.f;
        float total_sum_sq = 0.f;
        #pragma unroll
        for (int i = 0; i < num_warps; i++) {
            total_sum += warp_sum[i];
            total_sum_sq += warp_sum_sq[i];
        }
        mean = total_sum / num_elements;
        var = (total_sum_sq / num_elements) - (mean * mean);
        if (training) {
            running_mean[c] = (1.f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.f - momentum) * running_var[c] + momentum * var;
        }
        // Cache mean and variance in shared memory
        warp_sum[0] = mean;
        warp_sum[1] = var;
    }
    __syncthreads();

    // Retrieve computed mean and variance
    mean = warp_sum[0];
    var = warp_sum[1];
    float inv_std = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    // Normalize the input and write to output, unrolling the loop
    #pragma unroll
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
    // Launch one block per channel
    batch_norm_unroll_kernel<<<C, threads>>>(
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
    m.def("forward", &forward_cuda, "BatchNorm forward with loop unrolling (CUDA)");
}
