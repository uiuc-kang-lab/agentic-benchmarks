#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// Compute the KL divergence contribution per element
__device__ inline float compute_kldiv_value(float log_pred, float target) {
    return __expf(log_pred) - target * log_pred;
}

// Warp-level reduction using shuffle instructions
__device__ inline float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction across a warp using shared memory
__device__ inline float block_reduce_sum(float val, int tid, int block_size) {
    __shared__ float warp_sums[32];
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    
    // Each warp reduces its own value
    val = warp_reduce_sum(val);
    
    if (lane_id == 0) {
        warp_sums[warp_id] = val;
    }
    __syncthreads();
    
    // Let the first warp reduce the warp sums
    if (warp_id == 0) {
        val = (lane_id < (block_size + 31) / 32) ? warp_sums[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    __syncthreads();
    return val;
}

// Combined kernel: uses vectorized loads for groups of 4 elements and an efficient block reduction
__global__ void kl_div_kernel_combined(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;

    // Process complete groups of 4 elements via vectorized loads
    int vec_count = n / 4;  // number of complete float4 groups
    for (int i = gid; i < vec_count; i += stride) {
        // Load 4 floats at once from each array
        float4 log_vec = reinterpret_cast<const float4*>(log_predictions)[i];
        float4 target_vec = reinterpret_cast<const float4*>(targets)[i];

        local_sum += compute_kldiv_value(log_vec.x, target_vec.x)
                   + compute_kldiv_value(log_vec.y, target_vec.y)
                   + compute_kldiv_value(log_vec.z, target_vec.z)
                   + compute_kldiv_value(log_vec.w, target_vec.w);
    }

    // Block-level reduction using warp-level primitives
    local_sum = block_reduce_sum(local_sum, tid, blockDim.x);
    if (tid == 0) {
        atomicAdd(output, local_sum);
    }

    // Process tail elements (n not divisible by 4) using a single thread
    if (blockIdx.x == 0 && tid == 0) {
        int tail_start = vec_count * 4;
        float tail_sum = 0.0f;
        for (int j = tail_start; j < n; j++) {
            tail_sum += compute_kldiv_value(log_predictions[j], targets[j]);
        }
        atomicAdd(output, tail_sum);
    }
}

// Host wrapper for the combined kernel
torch::Tensor kl_div_cuda_forward_combined(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    int vec_count = n / 4;
    int blocks = (vec_count + threads - 1) / threads;
    if (blocks == 0) {
        blocks = 1;
    }
    blocks = std::min(blocks, 1024);

    kl_div_kernel_combined<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_combined, "Combined vectorized and reduced KLDiv forward (CUDA)");
}
