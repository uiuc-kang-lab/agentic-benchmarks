#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Warp-level reduction using shuffle intrinsic
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized kernel for KL divergence calculation using stride loops
__global__ void kl_div_kernel_stride_loops_opt(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {

    // Shared memory for partial sums - padded to avoid bank conflicts
    __shared__ float shared[33];  // 33 instead of 32 to avoid bank conflicts
    
    float local_sum = 0.0f;
    // Use grid-stride loop to evenly distribute the workload
    // Process multiple elements per thread to increase arithmetic intensity
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (; idx < n; idx += stride) {
        float lp = log_predictions[idx];
        float t = targets[idx];
        // Use FMA instruction for better performance
        local_sum = __fmaf_rn(t, -lp, local_sum);
        local_sum += expf(lp);
    }

    // Perform warp-level reduction
    local_sum = warpReduceSum(local_sum);

    // Each warp's lane 0 writes its partial sum into padded shared memory
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    // Final reduction of partial sums from each warp
    // Only threads in first warp participate
    if (threadIdx.x < warpSize) {
        float block_sum = 0.0f;
        int numWarps = (blockDim.x + warpSize - 1) / warpSize;
        if (threadIdx.x < numWarps) {
            block_sum = shared[threadIdx.x];
        }
        block_sum = warpReduceSum(block_sum);

        // Only thread 0 writes the final result
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward_stride_loops(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    kl_div_kernel_stride_loops_opt<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_stride_loops, "KL divergence forward using stride loops (CUDA optimized)");
}
