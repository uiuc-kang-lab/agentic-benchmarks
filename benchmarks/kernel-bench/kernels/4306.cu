#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// This kernel assigns one warp (32 threads) per channel, avoiding shared memory by using warp-level primitives
__global__ void batch_norm_warp_direct(
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

    // Each block processes one channel
    int c = blockIdx.x;
    int num_elements = N * H * W;
    int tid = threadIdx.x;  // blockDim.x is 32

    float sum = 0.0f;
    float sum_sq = 0.0f;

    // Each thread processes multiple elements in steps of warp size (32)
    for (int i = tid; i < num_elements; i += 32) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int index = ((n * C + c) * H + h) * W + w;
        float val = input[index];
        sum += val;
        sum_sq += val * val;
    }

    // Perform warp-level reduction using __shfl_down_sync
    unsigned int mask = 0xFFFFFFFF;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
        sum_sq += __shfl_down_sync(mask, sum_sq, offset);
    }

    float mean, var;
    // Lane 0 now contains the reductions for the warp
    if (tid == 0) {
        mean = sum / num_elements;
        var = sum_sq / num_elements - mean * mean;
        if (training) {
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var;
        } else {
            mean = running_mean[c];
            var = running_var[c];
        }
    }
    // Broadcast the computed mean and variance to all threads in the warp
    mean = __shfl_sync(mask, mean, 0);
    var  = __shfl_sync(mask, var,  0);
    
    float inv_std = rsqrtf(var + eps);
    float gamma = weight[c];
    float beta = bias[c];

    // Phase 2: Normalize input data
    for (int i = tid; i < num_elements; i += 32) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int index = ((n * C + c) * H + h) * W + w;
        float val = input[index];
        output[index] = (val - mean) * inv_std * gamma + beta;
    }
}

// Host function wrapping the kernel launch

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

    // Launch one block per channel with 32 threads per block (one warp per block)
    const int threads = 32;
    dim3 blocks(C);

    batch_norm_warp_direct<<<blocks, threads>>>(
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
    m.def("forward", &forward_cuda, "BatchNorm forward using warp-level primitives and no shared memory (CUDA)");
}
