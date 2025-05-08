#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Compute the KL divergence contribution per element
__device__ inline float compute_kldiv_value(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

// Warp-level reduction using shuffle instructions
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Optimized kernel with vectorized loads and efficient intra-block reduction
__global__ void kl_div_kernel_optimized_reduction(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int tid = threadIdx.x;
    int global_id = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    
    // Process in groups of 4 using vectorized loads
    int numVec = n / 4;  // number of complete float4 groups
    for (int i = global_id; i < numVec; i += stride) {
        float4 log_vec = reinterpret_cast<const float4*>(log_predictions)[i];
        float4 target_vec = reinterpret_cast<const float4*>(targets)[i];

        local_sum += compute_kldiv_value(log_vec.x, target_vec.x)
                   + compute_kldiv_value(log_vec.y, target_vec.y)
                   + compute_kldiv_value(log_vec.z, target_vec.z)
                   + compute_kldiv_value(log_vec.w, target_vec.w);
    }

    // Intra-warp reduction using warp-level primitives
    local_sum = warpReduceSum(local_sum);
    int lane = tid & 31;         // lane index within a warp
    int warp_id = tid >> 5;      // warp index within the block

    // Shared memory to store warp-level results
    __shared__ float shared_sum[32];
    if (lane == 0) {
        shared_sum[warp_id] = local_sum;
    }
    __syncthreads();

    float block_sum = 0.0f;
    // Let the first warp reduce the warp sums stored in shared memory
    int numWarps = (blockDim.x + 31) / 32;
    if (tid < numWarps) {
        block_sum = shared_sum[lane];
        block_sum = warpReduceSum(block_sum);
        if (lane == 0) {
            atomicAdd(output, block_sum);
        }
    }

    // Process tail elements (if n is not divisible by 4) using a single thread
    if (blockIdx.x == 0 && tid == 0) {
        float tail_sum = 0.0f;
        int tail_start = numVec * 4;
        for (int i = tail_start; i < n; i++) {
            tail_sum += compute_kldiv_value(log_predictions[i], targets[i]);
        }
        atomicAdd(output, tail_sum);
    }
}

// Host wrapper for the kernel
torch::Tensor kl_div_cuda_forward_optimized_reduction(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    int numVec = n / 4;
    int blocks = (numVec + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    blocks = std::min(blocks, 1024);

    kl_div_kernel_optimized_reduction<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_optimized_reduction, "Optimized reduction KLDiv forward (CUDA)");
}
