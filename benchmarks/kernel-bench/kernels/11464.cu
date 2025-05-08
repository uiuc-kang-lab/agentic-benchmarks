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

// Two-level reduction kernel for KL divergence calculation
__global__ void kl_div_kernel_hybrid(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {

    // Shared memory for partial sums
    extern __shared__ float shared[];
    
    float local_sum = 0.0f;
    
    // Grid-stride loop for coalesced memory access
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    #pragma unroll 4
    for (; idx < n; idx += stride) {
        float lp = __ldg(&log_predictions[idx]);
        float t = __ldg(&targets[idx]);
        local_sum += __expf(lp) - t * lp;  // Using faster intrinsic
    }

    // First level: Warp-level reduction using shuffle
    local_sum = warpReduceSum(local_sum);

    // Store warp results in shared memory
    int lane = threadIdx.x % warpSize;
    int warpId = threadIdx.x / warpSize;
    if (lane == 0) {
        shared[warpId] = local_sum;
    }
    __syncthreads();

    // Second level: Block-level reduction using shared memory
    if (threadIdx.x < (blockDim.x / warpSize)) {
        float warp_sum = shared[threadIdx.x];
        // Perform warp-level reduction on the partial sums
        warp_sum = warpReduceSum(warp_sum);
        
        if (threadIdx.x == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 512;  // Increased thread count
    const int blocks = min(65535, (n + threads - 1) / threads);
    const int shared_mem = (threads / 32) * sizeof(float);  // Space for warp results

    kl_div_kernel_hybrid<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA hybrid)");
}