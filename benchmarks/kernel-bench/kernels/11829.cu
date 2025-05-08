#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level reduction using shuffle operations without shared memory
__device__ inline float warp_reduce_sum(float val) {
    // Assumes warp size of 32
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel that uses only warp-level primitives for block-level reduction
// It processes data using vectorized loads (float4) to enhance memory throughput
// and each warp reduces its local sum and performs an atomicAdd, eliminating shared memory usage.
__global__ void kl_div_kernel_warp_only(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int lane = threadIdx.x & 31;

    float local_sum = 0.0f;

    // Process bulk data in groups of 4 using vectorized loads
    int vec_count = n / 4;  // number of complete float4 groups
    for (int i = global_tid; i < vec_count; i += stride) {
        // Load 4 floats at once
        float4 log_vec = reinterpret_cast<const float4*>(log_predictions)[i];
        float4 target_vec = reinterpret_cast<const float4*>(targets)[i];

        local_sum += __expf(log_vec.x) - target_vec.x * log_vec.x;
        local_sum += __expf(log_vec.y) - target_vec.y * log_vec.y;
        local_sum += __expf(log_vec.z) - target_vec.z * log_vec.z;
        local_sum += __expf(log_vec.w) - target_vec.w * log_vec.w;
    }

    // Reduce within each warp using only warp-level intrinsics
    local_sum = warp_reduce_sum(local_sum);
    if (lane == 0) {
        atomicAdd(output, local_sum);
    }

    // Handle tail elements (n not divisible by 4) by a single thread
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int tail_start = vec_count * 4;
        float tail_sum = 0.0f;
        for (int i = tail_start; i < n; i++) {
            float log_val = log_predictions[i];
            float target_val = targets[i];
            tail_sum += __expf(log_val) - target_val * log_val;
        }
        atomicAdd(output, tail_sum);
    }
}

// Host function to launch the kernel
torch::Tensor kl_div_cuda_forward_warp_only(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    // Use vectorized iterations count for grid calculation
    int vec_count = n / 4;
    int blocks = (vec_count + threads - 1) / threads;
    if (blocks < 1) blocks = 1;
    blocks = min(blocks, 1024);

    kl_div_kernel_warp_only<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_warp_only, "Warp-only KLDiv forward (CUDA)");
}
