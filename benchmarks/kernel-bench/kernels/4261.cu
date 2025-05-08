/*
 Fused BatchNorm Kernel using Cooperative Groups
 Combines reduction (for mean/variance computation) and normalization into a single kernel launch.
 This kernel uses atomic operations and grid-wide synchronization via cooperative groups to fuse the three-phase algorithm
 (partial reduction, final reduction, normalization) into one kernel, reducing kernel launch overhead.

 IMPORTANT: This kernel must be launched as a cooperative kernel (using cudaLaunchCooperativeKernel or similar) so that grid-sync works.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Warp-level reduction using shuffle intrinsics
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using warp-level primitives and shared memory
__inline__ __device__ float blockReduceSum(float val) {
    __shared__ float shared[32]; // one value per warp
    int lane = threadIdx.x & (warpSize - 1);
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);

    if (lane == 0) {
        shared[wid] = val;
    }
    __syncthreads();

    // Only the first warp participates
    val = (threadIdx.x < (blockDim.x/warpSize)) ? shared[lane] : 0.0f;
    if (wid == 0) {
        val = warpReduceSum(val);
    }
    return val;
}

/*
 fused_batch_norm_kernel
 
 Parameters:
   input        - pointer to input tensor (N, C, H, W) in row-major order
   weight       - scale per channel
   bias         - bias per channel
   running_mean - running mean (updated during training)
   running_var  - running variance (updated during training)
   training     - flag indicating training mode
   momentum     - momentum for running stats
   eps          - epsilon for numerical stability
   output       - pointer to output tensor (same shape as input)
   globalSum    - temporary global buffer of size C (must be preinitialized to 0)
   globalSumSq  - temporary global buffer of size C (must be preinitialized to 0)
   computedMean - output buffer for computed mean per channel (size: C)
   computedVar  - output buffer for computed variance per channel (size: C)
   N, C, H, W   - dimensions of the input
 
 Grid configuration:
   grid.x = C (one slice per channel)
   grid.y = blocksPerChannel (multiple blocks per channel for reduction and normalization)
   blockDim.x = number of threads (e.g., 256)

 Note: This kernel must be launched as a cooperative kernel to allow grid-wide synchronization.
*/

__global__ void fused_batch_norm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    bool training,
    float momentum,
    float eps,
    float* __restrict__ output,
    float* __restrict__ globalSum,    // size: C, preinitialized to 0
    float* __restrict__ globalSumSq,  // size: C, preinitialized to 0
    float* __restrict__ computedMean, // size: C
    float* __restrict__ computedVar,  // size: C
    int N, int C, int H, int W
){
    // Determine channel and per-channel element count
    int c = blockIdx.x; // each grid.x corresponds to one channel
    int blocksPerChannel = gridDim.y;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    int numElements = N * H * W;

    // Phase 1: Partial Reduction for this channel
    // Each block processes a disjoint slice of the channel elements.
    // Compute starting offset and stride that covers all elements in the channel.
    int start = blockIdx.y * blockSize + tid; // unique starting index per block within channel
    int stride = blockSize * blocksPerChannel; // stride over blocks in the channel

    float sum = 0.0f;
    float sumSq = 0.0f;

    for (int i = start; i < numElements; i += stride) {
        // Compute n, h, w given flat index i in the channel
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        sum += val;
        sumSq += val * val;
    }

    // Within the block, reduce the partial sums
    sum = blockReduceSum(sum);
    sumSq = blockReduceSum(sumSq);

    // Thread 0 of each block accumulates the block's result into global buffers
    if(tid == 0) {
        atomicAdd(&globalSum[c], sum);
        atomicAdd(&globalSumSq[c], sumSq);
    }

    // Ensure all blocks complete phase 1
    cg::grid_group grid = cg::this_grid();
    grid.sync();

    // Phase 2: Final reduction and computation of mean and variance
    // Let one designated thread per channel (blockIdx.y == 0, tid == 0) perform final computation
    if (blockIdx.y == 0 && tid == 0) {
        float mean = globalSum[c] / numElements;
        float var = globalSumSq[c] / numElements - mean * mean;
        if (training) {
            // Update running averages
            running_mean[c] = (1 - momentum) * running_mean[c] + momentum * mean;
            running_var[c]  = (1 - momentum) * running_var[c]  + momentum * var;
            computedMean[c] = mean;
            computedVar[c]  = var;
        } else {
            // In eval, use stored running stats
            computedMean[c] = running_mean[c];
            computedVar[c]  = running_var[c];
        }
    }

    // Ensure all threads see the computed mean/variance
    grid.sync();

    // Phase 3: Normalization
    float mean = computedMean[c];
    float var = computedVar[c];
    float invStd = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    // Each block normalizes its assigned slice of the channel
    for (int i = start; i < numElements; i += stride) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        int idx = ((n * C + c) * H + h) * W + w;
        float val = input[idx];
        output[idx] = (val - mean) * invStd * w_val + b_val;
    }
}

// Host function to launch the fused kernel
// This function allocates temporary buffers for reduction and launches the cooperative kernel.

torch::Tensor fused_forward_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor running_mean,
    torch::Tensor running_var,
    bool training,
    float momentum,
    float eps
) {
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
    int numChannelElements = N * H * W;

    auto output = torch::empty_like(input);
    auto options = torch::TensorOptions().dtype(input.dtype()).device(input.device());

    // Temporary buffers for global accumulation and storing computed stats (size: C)
    auto globalSum = torch::zeros({C}, options);
    auto globalSumSq = torch::zeros({C}, options);
    auto computedMean = torch::empty({C}, options);
    auto computedVar = torch::empty({C}, options);

    // Kernel configuration
    int threads = 256;
    // Compute how many blocks per channel are needed to process all elements
    int blocksPerChannel = (numChannelElements + threads - 1) / threads;
    // Grid: one block per channel in x dimension, and blocksPerChannel in y dimension
    dim3 grid(C, blocksPerChannel);

    /*
     IMPORTANT: This kernel uses grid-wide synchronization (cooperative groups).
     It must be launched as a cooperative kernel. The launching mechanism (e.g. cudaLaunchCooperativeKernel)
     must be used and the device must support cooperative launch.
    */

    fused_batch_norm_kernel<<<grid, threads>>>(
        input.data_ptr<float>(),
        weight.data_ptr<float>(),
        bias.data_ptr<float>(),
        running_mean.data_ptr<float>(),
        running_var.data_ptr<float>(),
        training,
        momentum,
        eps,
        output.data_ptr<float>(),
        globalSum.data_ptr<float>(),
        globalSumSq.data_ptr<float>(),
        computedMean.data_ptr<float>(),
        computedVar.data_ptr<float>(),
        N, C, H, W
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_forward_cuda, "Fused BatchNorm forward (CUDA, cooperative kernel)");
}
