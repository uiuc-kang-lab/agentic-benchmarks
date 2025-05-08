#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Device function to compute partial sums with 128-bit aligned, read-only loads using __ldg().
__device__ void computePartialSums(const float* __restrict__ input,
                                     int c, int N, int C, int H, int W,
                                     int tid, int stride,
                                     float &partialSum, float &partialSumSq) {
    int numElements = N * H * W;
    partialSum = 0.0f;
    partialSumSq = 0.0f;
    
    // Loop over the channel's elements
    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        // Use __ldg() for read-only load which is expected to be 128-bit aligned
        float val = __ldg(&input[idx]);
        partialSum += val;
        partialSumSq += val * val;
    }
}

// Block-level reduction using warp-level primitives and shared memory
__device__ void blockReduceSum(float &sum, float &sumSq) {
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;

    float sum_val = warpReduceSum(sum);
    float sumSq_val = warpReduceSum(sumSq);

    __shared__ float sharedSum[32];
    __shared__ float sharedSumSq[32];

    if(lane == 0) {
        sharedSum[warpId] = sum_val;
        sharedSumSq[warpId] = sumSq_val;
    }
    __syncthreads();

    if(threadIdx.x < warpSize) {
        float s = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? sharedSum[threadIdx.x] : 0.0f;
        float s_sq = (threadIdx.x < (blockDim.x + warpSize - 1) / warpSize) ? sharedSumSq[threadIdx.x] : 0.0f;
        s = warpReduceSum(s);
        s_sq = warpReduceSum(s_sq);
        if(threadIdx.x == 0) {
            sum = s;
            sumSq = s_sq;
        }
    }
    __syncthreads();
}

// Device function to normalize a value using computed mean, invStd, weight and bias
__device__ inline float normalizeValue(float val, float mean, float invStd, float w, float b) {
    return (val - mean) * invStd * w + b;
}

// Kernel that performs BatchNorm using __ldg() for optimized read-only global memory loads.
// It assumes that the input, weight, and bias tensors are 128-bit aligned.
__global__ void aligned_ldg_batch_norm_kernel(
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
    int tid = threadIdx.x;
    int stride = blockDim.x;
    int numElements = N * H * W;

    float partialSum, partialSumSq;
    computePartialSums(input, c, N, C, H, W, tid, stride, partialSum, partialSumSq);
    blockReduceSum(partialSum, partialSumSq);

    __shared__ float stats[2]; // stats[0]: mean, stats[1]: var
    float mean, var;
    if (tid == 0) {
        mean = partialSum / numElements;
        var = partialSumSq / numElements - mean * mean;
        if (training) {
            // Update running statistics
            running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
            running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var;
        } else {
            // Use read-only __ldg() to fetch precomputed statistics
            mean = __ldg(&running_mean[c]);
            var = __ldg(&running_var[c]);
        }
        stats[0] = mean;
        stats[1] = var;
    }
    __syncthreads();
    mean = stats[0];
    var = stats[1];

    float invStd = rsqrtf(var + eps);
    // Use __ldg() to load weight and bias from global memory
    float channelWeight = __ldg(&weight[c]);
    float channelBias   = __ldg(&bias[c]);

    // Normalize input and write output; using __ldg() for read access on input
    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = __ldg(&input[idx]);
        output[idx] = normalizeValue(val, mean, invStd, channelWeight, channelBias);
    }
}

// Host function to launch the kernel from PyTorch
torch::Tensor aligned_ldg_forward_cuda(
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
    int threads = 512;
    
    // Launch one block per channel
    aligned_ldg_batch_norm_kernel<<<C, threads>>>(
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
    m.def("forward", &aligned_ldg_forward_cuda, "Aligned LDG BatchNorm forward (CUDA)");
}
