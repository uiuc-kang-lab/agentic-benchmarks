#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Device function to compute sum and sum of squares
__device__ void compute_sum_and_sum_sq(const float* input, int num_elements, int stride, int tid, int C, int H, int W, int c, float& sum, float& sum_sq) {
    sum = 0.0f;
    sum_sq = 0.0f;
    for (int i = tid; i < num_elements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w = hw % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }
}

// Device function to perform warp-level reduction
__device__ void warp_reduce(float& sum, float& sum_sq) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
        sum_sq += __shfl_down_sync(0xffffffff, sum_sq, offset);
    }
}

// Device function to compute mean and variance
__device__ void compute_mean_variance(float& mean, float& var, float total_sum, float total_sum_sq, int num_elements, float momentum, float* running_mean, float* running_var, int c, bool training) {
    mean = total_sum / num_elements;
    var = (total_sum_sq / num_elements) - (mean * mean);
    if (training) {
        running_mean[c] = (1.f - momentum) * running_mean[c] + momentum * mean;
        running_var[c] = (1.f - momentum) * running_var[c] + momentum * var;
    }
}

// Device function to normalize input and write to output
__device__ void normalize_and_write_output(const float* input, float* output, int num_elements, int stride, int tid, int C, int H, int W, int c, float mean, float inv_std, float w_val, float b_val) {
    for (int i = tid; i < num_elements; i += stride) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w_idx = hw % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float val = input[idx];
        output[idx] = (val - mean) * inv_std * w_val + b_val;
    }
}

// Main kernel function
__global__ void batch_norm_modular_kernel(
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

    float my_sum, my_sum_sq;
    compute_sum_and_sum_sq(input, num_elements, blockSize, tid, C, H, W, c, my_sum, my_sum_sq);

    warp_reduce(my_sum, my_sum_sq);

    __shared__ float warp_sum[32];
    __shared__ float warp_sum_sq[32];
    int warpId = tid / warpSize;
    int lane = tid % warpSize;
    if (lane < 32) {
        warp_sum[warpId] = my_sum;
        warp_sum_sq[warpId] = my_sum_sq;
    }
    __syncthreads();

    float mean = 0.f, var = 0.f;
    if (tid == 0) {
        int num_warps = blockSize / warpSize;
        float total_sum = 0.f;
        float total_sum_sq = 0.f;
        for (int i = 0; i < num_warps; i++) {
            total_sum += warp_sum[i];
            total_sum_sq += warp_sum_sq[i];
        }
        compute_mean_variance(mean, var, total_sum, total_sum_sq, num_elements, momentum, running_mean, running_var, c, training);
        warp_sum[0] = mean;
        warp_sum[1] = var;
    }
    __syncthreads();

    mean = warp_sum[0];
    var = warp_sum[1];
    float inv_std = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    normalize_and_write_output(input, output, num_elements, blockSize, tid, C, H, W, c, mean, inv_std, w_val, b_val);
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

    // Input validity checks
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
    batch_norm_modular_kernel<<<C, threads>>>(
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
    m.def("forward", &forward_cuda, "Modular BatchNorm forward (CUDA)");
}
