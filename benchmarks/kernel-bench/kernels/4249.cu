#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel to compute BatchNorm with uniform control flow to minimize warp divergence
__global__ void uniform_branch_batchnorm_kernel(
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
    int stride = blockDim.x;
    int numElements = N * H * W;

    // Phase 1: Compute partial sum and sum of squares
    float partialSum = 0.0f;
    float partialSumSq = 0.0f;
    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int r = i % (H * W);
        int h = r / W;
        int w = r % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        partialSum += val;
        partialSumSq += val * val;
    }

    // Perform warp-level reduction
    partialSum = warpReduceSum(partialSum);
    partialSumSq = warpReduceSum(partialSumSq);

    // Use shared memory to reduce across warps
    __shared__ float sharedWarpSum[32];  // supports up to 32 warps per block
    __shared__ float sharedWarpSumSq[32];
    int lane = tid % warpSize;
    int warpId = tid / warpSize;
    if(lane == 0) {
        sharedWarpSum[warpId] = partialSum;
        sharedWarpSumSq[warpId] = partialSumSq;
    }
    __syncthreads();

    float totalSum = 0.0f;
    float totalSumSq = 0.0f;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if(tid < numWarps) {
        totalSum = sharedWarpSum[tid];
        totalSumSq = sharedWarpSumSq[tid];
    }
    if(tid < warpSize) {
        totalSum = warpReduceSum(totalSum);
        totalSumSq = warpReduceSum(totalSumSq);
    }

    // Phase 2: Compute statistics without divergent branches
    // All threads compute computed_mean and computed_var, but only thread 0 uses them
    float computed_mean = totalSum / numElements;
    float computed_var = totalSumSq / numElements - computed_mean * computed_mean;

    // Convert training flag to float (1.0 if true, 0.0 if false) to avoid branching
    float flag = static_cast<float>(training);
    // If training, update running stats; if not, new_mean/var equals the running stats
    float new_mean = (1.0f - flag * momentum) * running_mean[c] + flag * momentum * computed_mean;
    float new_var  = (1.0f - flag * momentum) * running_var[c]  + flag * momentum * computed_var;

    __shared__ float stats[2];  // stats[0] = mean, stats[1] = var
    if (tid == 0) {
        // In inference mode (training==false, flag==0), this writes the same value back
        running_mean[c] = new_mean;
        running_var[c] = new_var;
        stats[0] = new_mean;
        stats[1] = new_var;
    }
    __syncthreads();
    float mean = stats[0];
    float var  = stats[1];
    
    // Phase 3: Normalize the input
    float invStd = rsqrtf(var + eps);
    float channelWeight = weight[c];
    float channelBias = bias[c];
    
    for (int i = tid; i < numElements; i += stride) {
        int n = i / (H * W);
        int r = i % (H * W);
        int h = r / W;
        int w = r % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        output[idx] = (val - mean) * invStd * channelWeight + channelBias;
    }
}

// Host function exposed to PyTorch
torch::Tensor uniform_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps) {

    // Input validations
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

    int threads = 256;
    // No dynamic shared memory is needed since shared arrays are statically allocated
    uniform_branch_batchnorm_kernel<<<C, threads>>>(
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
    m.def("forward", &uniform_forward_cuda, "BatchNorm forward with minimal warp divergence (CUDA)");
}
