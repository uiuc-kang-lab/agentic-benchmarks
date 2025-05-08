#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Templated kernel implementing BatchNorm with a tunable block size
template <int BLOCK_SIZE>
__global__ void adaptive_template_batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    bool training,
    float momentum,
    float eps,
    float* __restrict__ output,
    int N, int C, int H, int W) {

    // Each block processes one channel
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int numElements = N * H * W;
    int stride = BLOCK_SIZE;

    // Phase 1: Compute partial sums for mean and variance
    float partialSum = 0.0f, partialSumSq = 0.0f;
    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        partialSum += val;
        partialSumSq += val * val;
    }

    __shared__ float sharedSum[BLOCK_SIZE];
    __shared__ float sharedSumSq[BLOCK_SIZE];
    sharedSum[tid] = partialSum;
    sharedSumSq[tid] = partialSumSq;
    __syncthreads();

    // Reduce within block
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
            sharedSumSq[tid] += sharedSumSq[tid + s];
        }
        __syncthreads();
    }

    __shared__ float mean;
    __shared__ float var;
    if (tid == 0) {
        mean = sharedSum[0] / numElements;
        var = sharedSumSq[0] / numElements - mean * mean;
        if (training) {
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
    }
    __syncthreads();

    float invStd = rsqrtf(var + eps);
    float channelWeight = weight[c];
    float channelBias = bias[c];

    // Phase 2: Normalize
    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        output[idx] = (val - mean) * invStd * channelWeight + channelBias;
    }
}

// Host function that selects an optimal block size from a set of candidates
torch::Tensor adaptive_template_forward_cuda(
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

    int N = input.size(0);
    int C = input.size(1);
    int H = input.size(2);
    int W = input.size(3);
    auto output = torch::empty_like(input);

    int numElements = N * H * W;
    int block_size;
    // Heuristic: choose block size based on the number of elements per channel
    if (numElements <= 512)
        block_size = 32;
    else if (numElements <= 1024)
        block_size = 64;
    else if (numElements <= 4096)
        block_size = 128;
    else if (numElements <= 8192)
        block_size = 256;
    else
        block_size = 512;

    // Each channel is processed by one block
    dim3 grid(C);

    switch (block_size) {
        case 32:
            adaptive_template_batch_norm_kernel<32><<<grid, 32>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(),
                training,
                momentum,
                eps,
                output.data_ptr<float>(),
                N, C, H, W);
            break;
        case 64:
            adaptive_template_batch_norm_kernel<64><<<grid, 64>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(),
                training,
                momentum,
                eps,
                output.data_ptr<float>(),
                N, C, H, W);
            break;
        case 128:
            adaptive_template_batch_norm_kernel<128><<<grid, 128>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(),
                training,
                momentum,
                eps,
                output.data_ptr<float>(),
                N, C, H, W);
            break;
        case 256:
            adaptive_template_batch_norm_kernel<256><<<grid, 256>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(),
                training,
                momentum,
                eps,
                output.data_ptr<float>(),
                N, C, H, W);
            break;
        case 512:
            adaptive_template_batch_norm_kernel<512><<<grid, 512>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(),
                training,
                momentum,
                eps,
                output.data_ptr<float>(),
                N, C, H, W);
            break;
        default:
            // Fallback option
            adaptive_template_batch_norm_kernel<256><<<grid, 256>>>(
                input.data_ptr<float>(),
                weight.data_ptr<float>(),
                bias.data_ptr<float>(),
                running_mean.data_ptr<float>(),
                running_var.data_ptr<float>(),
                training,
                momentum,
                eps,
                output.data_ptr<float>(),
                N, C, H, W);
            break;
    }

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &adaptive_template_forward_cuda, "Adaptive Template BatchNorm forward (CUDA)");
}
