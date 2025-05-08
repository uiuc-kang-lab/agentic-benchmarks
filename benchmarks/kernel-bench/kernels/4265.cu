#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// Kernel that applies BatchNorm with improved memory coalescing using vectorized loads/stores
__global__ void coalesced_batchnorm_kernel(
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

    // Each block processes one channel (c)
    int channel_elems = N * H * W;
    int c = blockIdx.x; // channel index
    const float* channel_ptr = input + c * channel_elems;
    float* out_channel_ptr = output + c * channel_elems;

    // Compute number of elements that can be processed with vectorized load (float4)
    int vec_count = channel_elems / 4;            // number of float4 elements
    int remainder = channel_elems - vec_count * 4;  // remaining elements to process

    // --- Phase 1: Compute mean and variance ---
    // Use vectorized loads for coalesced memory access
    float partialSum = 0.0f;
    float partialSumSq = 0.0f;
    const float4* in_vec = reinterpret_cast<const float4*>(channel_ptr);

    // Each thread processes consecutive float4 elements, ensuring coalesced loads
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        float4 v = in_vec[i];
        partialSum   += v.x + v.y + v.z + v.w;
        partialSumSq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }

    // Process any remaining elements with scalar loads
    int rem_start = vec_count * 4;
    for (int i = threadIdx.x; i < remainder; i += blockDim.x) {
        int idx = rem_start + i;
        float val = channel_ptr[idx];
        partialSum   += val;
        partialSumSq += val * val;
    }

    // Warp-level reduction using shuffle intrinsics
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        partialSum   += __shfl_down_sync(mask, partialSum, offset);
        partialSumSq += __shfl_down_sync(mask, partialSumSq, offset);
    }

    // Use shared memory to aggregate results from each warp
    __shared__ float sharedSum[64];     // enough for up to 64 warps
    __shared__ float sharedSumSq[64];
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        sharedSum[warpId] = partialSum;
        sharedSumSq[warpId] = partialSumSq;
    }
    __syncthreads();

    // First warp reduces the partial sums from each warp
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;
    if (threadIdx.x < numWarps) {
        partialSum = sharedSum[threadIdx.x];
        partialSumSq = sharedSumSq[threadIdx.x];
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            partialSum   += __shfl_down_sync(mask, partialSum, offset);
            partialSumSq += __shfl_down_sync(mask, partialSumSq, offset);
        }
        if (threadIdx.x == 0) {
            float mean = partialSum / channel_elems;
            float var  = partialSumSq / channel_elems - mean * mean;
            if (training) {
                running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * mean;
                running_var[c]  = (1.0f - momentum) * running_var[c]  + momentum * var;
            } else {
                mean = running_mean[c];
                var  = running_var[c];
            }
            // Store stats in shared memory so all threads can access them
            sharedSum[0] = mean;
            sharedSumSq[0] = var;
        }
    }
    __syncthreads();

    float mean = sharedSum[0];
    float var  = sharedSumSq[0];
    float invStd = rsqrtf(var + eps);
    float w_val = weight[c];
    float b_val = bias[c];

    // --- Phase 2: Normalize the input using computed mean and variance ---
    // Use vectorized operations for coalesced global memory write
    float4* out_vec = reinterpret_cast<float4*>(out_channel_ptr);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
        float4 v = in_vec[i]; // coalesced read
        v.x = (v.x - mean) * invStd * w_val + b_val;
        v.y = (v.y - mean) * invStd * w_val + b_val;
        v.z = (v.z - mean) * invStd * w_val + b_val;
        v.w = (v.w - mean) * invStd * w_val + b_val;
        out_vec[i] = v; // coalesced write
    }
    // Process remaining elements with scalar operations
    for (int i = threadIdx.x; i < remainder; i += blockDim.x) {
        int idx = vec_count * 4 + i;
        float val = channel_ptr[idx];
        out_channel_ptr[idx] = (val - mean) * invStd * w_val + b_val;
    }
}

// Host function called from PyTorch
torch::Tensor coalesced_forward_cuda(
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
    
    // Launch one block per channel
    int threads = 256;
    dim3 grid(C);
    coalesced_batchnorm_kernel<<<grid, threads>>>(
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
    m.def("forward", &coalesced_forward_cuda, "Coalesced BatchNorm forward (CUDA)");
}
