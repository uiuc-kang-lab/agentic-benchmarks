#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel Stage 1: Each block computes a partial sum without using atomic operations
// and writes its result in a dedicated global array slot.
__global__ void kl_div_kernel_stage1(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ partial_sums,
    const int n) {

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    float sum = 0.0f;

    // Use vectorized loads with float4 for aligned accesses
    int n4 = n / 4;  // number of vectorized groups
    const float4* logp_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);
    int stride = blockDim.x * gridDim.x;
    int vec_idx = global_idx;
    
    while (vec_idx < n4) {
        float4 lp = __ldg(&logp_vec[vec_idx]);
        float4 tt = __ldg(&targ_vec[vec_idx]);
        sum += expf(lp.x) - tt.x * lp.x
            + expf(lp.y) - tt.y * lp.y
            + expf(lp.z) - tt.z * lp.z
            + expf(lp.w) - tt.w * lp.w;
        vec_idx += stride;
    }

    // Process remainder elements using scalar loads
    int remaining_start = n4 * 4;
    int scalar_idx = remaining_start + global_idx;
    while (scalar_idx < n) {
        float lp = __ldg(&log_predictions[scalar_idx]);
        float tt = __ldg(&targets[scalar_idx]);
        sum += expf(lp) - tt * lp;
        scalar_idx += stride;
    }

    // Reduction in shared memory
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write block's result to global partial sum array
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel Stage 2: Reduce the partial sums array into a single sum using one block
__global__ void kl_div_kernel_stage2(
    const float* __restrict__ partial_sums,
    float* __restrict__ output,
    const int num_elements) {

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = tid;
    float sum = 0.0f;
    int stride = blockDim.x;

    // Load partial sums into shared memory
    while (idx < num_elements) {
        sum += partial_sums[idx];
        idx += stride;
    }
    sdata[tid] = sum;
    __syncthreads();

    // Shared-memory reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[0] = sdata[0];
    }
}

// Host function to perform two-stage reduction for the KL divergence computation
// This approach eliminates the use of atomic operations in global memory during stage 1
// by writing block-local results to a dedicated output array and then reducing that array.

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::empty({1}, log_predictions.options());

    // Stage 1 configuration
    const int threads = 256;
    int blocks = (n + threads * 4 - 1) / (threads * 4);
    // Cap number of blocks if necessary
    blocks = min(blocks, 1024);
    auto partial_sum_tensor = torch::empty({blocks}, log_predictions.options());

    int shared_mem_size = threads * sizeof(float);
    kl_div_kernel_stage1<<<blocks, threads, shared_mem_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        partial_sum_tensor.data_ptr<float>(),
        n);

    // Stage 2 configuration: Use one block to reduce the partial sums
    int threads2 = 256;
    int shared_mem_size2 = threads2 * sizeof(float);
    kl_div_kernel_stage2<<<1, threads2, shared_mem_size2>>>(
        partial_sum_tensor.data_ptr<float>(),
        output.data_ptr<float>(),
        blocks);

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA Two-Stage Reduction, Minimal Atomics)");
}
