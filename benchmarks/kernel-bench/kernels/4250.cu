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

// Kernel implementing batch normalization with optimized reduction
// using shared memory for intra-warp results and warp-level primitives for final reduction
__global__ void warp_shared_red_batch_norm_kernel(
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

    // Each block handles one channel
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int numElements = N * H * W;

    // Shared memory arrays for per-warp partial sums
    __shared__ float warpSums[32];       // supports up to 32 warps per block
    __shared__ float warpSumsSq[32];
    __shared__ float stats[2];           // [0]: mean, [1]: variance

    float mean, var;

    if (training) {
        float sum = 0.0f;
        float sumSq = 0.0f;
        // Phase 1: Each thread computes a partial sum and sum of squares
        for (int i = tid; i < numElements; i += blockSize) {
            int n = i / (H * W);
            int rem = i % (H * W);
            int h = rem / W;
            int w = rem % W;
            int idx = ((n * C + c) * H + h) * W + w;
            float val = input[idx];
            sum += val;
            sumSq += val * val;
        }

        // Intra-warp reduction using warp-level primitives
        sum = warpReduceSum(sum);
        sumSq = warpReduceSum(sumSq);

        // Each warp leader writes its result to shared memory
        if ((tid & (warpSize - 1)) == 0) {
            int warpId = tid / warpSize;
            warpSums[warpId] = sum;
            warpSumsSq[warpId] = sumSq;
        }
        __syncthreads();

        // Final reduction by thread 0 across warp results
        if (tid == 0) {
            int numWarps = (blockSize + warpSize - 1) / warpSize;
            float totalSum = 0.0f;
            float totalSumSq = 0.0f;
            for (int i = 0; i < numWarps; i++) {
                totalSum += warpSums[i];
                totalSumSq += warpSumsSq[i];
            }
            mean = totalSum / numElements;
            var = totalSumSq / numElements - mean * mean;
            // Update running statistics
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c] = (1 - momentum) * running_var[c] + momentum * var;
            stats[0] = mean;
            stats[1] = var;
        }
        __syncthreads();
    } else {
        // In inference mode, simply use running statistics
        if (tid == 0) {
            stats[0] = running_mean[c];
            stats[1] = running_var[c];
        }
        __syncthreads();
    }
    
    mean = stats[0];
    var = stats[1];
    float invStd = rsqrtf(var + eps);
    float scale = weight[c];
    float shift = bias[c];

    // Phase 2: Normalize each element and write to output
    for (int i = tid; i < numElements; i += blockSize) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float in_val = input[idx];
        output[idx] = (in_val - mean) * invStd * scale + shift;
    }
}

// Host function exposed to PyTorch
torch::Tensor warp_shared_red_forward_cuda(
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
    int threads = 256;
    size_t shared_mem = 0; // Using statically allocated shared memory arrays

    // Launch one block per channel
    warp_shared_red_batch_norm_kernel<<<C, threads, shared_mem>>>(
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
    m.def("forward", &warp_shared_red_forward_cuda, "Warp and shared memory reduction BatchNorm forward (CUDA)");
}
