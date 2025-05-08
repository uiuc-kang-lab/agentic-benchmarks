#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level reduction using shuffle instructions without divergent branching
__device__ inline float warp_reduce_sum(float val) {
    // Unroll the warp reduction (assuming warp size is 32)
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// CUDA kernel implementing KL divergence with minimized warp divergence by refactoring conditionals
// to use arithmetic masks in the warp reduction stages.
__global__ void kl_div_no_divergence_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Use vectorized loads for coalesced memory access
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int vec_size = 4;
    const int vec_count = n / vec_size;

    float sum = 0.0f;

    // Process vectorized elements
    for (int i = tid; i < vec_count; i += stride) {
        float4 log_vec = reinterpret_cast<const float4*>(log_predictions)[i];
        float4 tgt_vec = reinterpret_cast<const float4*>(targets)[i];
        sum += expf(log_vec.x) - tgt_vec.x * log_vec.x;
        sum += expf(log_vec.y) - tgt_vec.y * log_vec.y;
        sum += expf(log_vec.z) - tgt_vec.z * log_vec.z;
        sum += expf(log_vec.w) - tgt_vec.w * log_vec.w;
    }

    // Process remaining tail elements
    int processed = vec_count * vec_size;
    for (int i = processed + tid; i < n; i += stride) {
        float log_val = log_predictions[i];
        float tgt_val = targets[i];
        sum += expf(log_val) - tgt_val * log_val;
    }

    // Perform warp-level reduction
    float warp_sum = warp_reduce_sum(sum);

    // Shared memory to hold one reduced value per warp
    extern __shared__ float shared[];

    // Instead of using divergent branches to select warp leaders, we use a ternary operator
    // which is typically compiled to a predicated (branchless) select instruction.
    // Each thread computes a flag that is 1.0f if it is the warp leader (lane 0), 0.0f otherwise.
    float is_warp_leader = (threadIdx.x % 32 == 0) ? 1.0f : 0.0f;

    // Each warp leader should write its warp_sum to shared memory. Using the arithmetic mask ensures
    // that non-leader threads contribute 0.0f without divergent control flow.
    // First, initialize shared memory for this warp. Only one thread per warp (leader) writes 0.0f.
    // Although the ternary operator is used, it is resolved with a predicated instruction.
    if ((threadIdx.x % 32) == 0) {
        shared[threadIdx.x / 32] = 0.0f;
    }
    __syncthreads();

    // Each thread performs an atomicAdd into its warp's slot with its contribution multiplied by the leader flag
    // so that only the warp leader effectively adds its warp_sum.
    int warp_id = threadIdx.x / 32;
    atomicAdd(&shared[warp_id], warp_sum * is_warp_leader);
    __syncthreads();

    // Now, threads in the first warp (warp_id 0) reduce the warp sums stored in shared memory
    float block_sum = 0.0f;
    // Use a flag that is 1.0f for threads in lane < (blockDim.x/32) (i.e. within the number of warps in the block), else 0
    int num_warps = blockDim.x / 32;
    float warp_val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    float total = warp_reduce_sum(warp_val);
    block_sum = total;

    // Finally, only one thread (using a branchless flag) adds the block_sum to the global output.
    float is_global_leader = (threadIdx.x == 0) ? 1.0f : 0.0f;
    atomicAdd(output, block_sum * is_global_leader);
}

// Host function to launch the kernel
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Launch configuration: tuning blocks and threads for the GPU
    const int threads = 256;
    const int blocks = 128;
    // Shared memory size: one float per warp
    const int shared_mem = (threads / 32) * sizeof(float);

    kl_div_no_divergence_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with minimal warp divergence using branchless arithmetic (CUDA)");
}
