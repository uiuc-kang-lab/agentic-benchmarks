#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define WARP_SIZE 32

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Combined kernel that uses warp-level reduction for mean/variance computation
// and coalesced memory accesses for normalization.
__global__ void combined_batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    const bool training,
    const float momentum,
    const float eps,
    float* __restrict__ output,
    const int N,
    const int C,
    const int H,
    const int W) {

    // Each block is responsible for one channel
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    const int num_elements = N * H * W;

    // Determine warp information
    const int warpId = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;
    const int warpsPerBlock = blockDim.x / WARP_SIZE;

    // Shared memory for storing warp-level sums
    __shared__ float warpSums[32];    // assuming at most 32 warps per block
    __shared__ float warpSumSqs[32];
    __shared__ float channelMean;
    __shared__ float channelInvStd;

    // Phase 1: Compute partial sums for mean and variance
    float sum = 0.0f;
    float sumSq = 0.0f;
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w_idx = hw % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float val = input[idx];
        sum += val;
        sumSq += val * val;
    }

    // Warp-level reduction for sum and sum of squares
    sum = warpReduceSum(sum);
    sumSq = warpReduceSum(sumSq);

    // Each warp writes its result to shared memory
    if (lane == 0) {
        warpSums[warpId] = sum;
        warpSumSqs[warpId] = sumSq;
    }
    __syncthreads();

    // Final reduction by the first warp
    if (warpId == 0) {
        sum = (lane < warpsPerBlock) ? warpSums[lane] : 0.0f;
        sumSq = (lane < warpsPerBlock) ? warpSumSqs[lane] : 0.0f;
        sum = warpReduceSum(sum);
        sumSq = warpReduceSum(sumSq);

        float mean = sum / num_elements;
        float var = (sumSq / num_elements) - (mean * mean);
        
        // Update running statistics if in training mode
        if (training) {
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
        } else {
            // Use stored running statistics during inference
            mean = running_mean[c];
            var = running_var[c];
        }
        channelMean = mean;
        channelInvStd = rsqrtf(var + eps);
    }
    __syncthreads();

    // Phase 2: Normalize the input using the computed mean and variance
    float mean = channelMean;
    float invStd = channelInvStd;
    float w_val = weight[c];
    float b_val = bias[c];

    #pragma unroll 4
    for (int i = tid; i < num_elements; i += blockDim.x) {
        int n = i / (H * W);
        int hw = i % (H * W);
        int h = hw / W;
        int w_idx = hw % W;
        int idx = ((n * C + c) * H + h) * W + w_idx;
        float val = input[idx];
        output[idx] = (val - mean) * invStd * w_val + b_val;
    }
}

// Host wrapper function
torch::Tensor combined_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {
    
    // Validity checks
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
    combined_batch_norm_kernel<<<C, threads>>>(
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
    m.def("forward", &combined_forward_cuda, "Combined BatchNorm forward (CUDA)");
}
