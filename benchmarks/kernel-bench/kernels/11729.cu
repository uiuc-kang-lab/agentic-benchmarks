/*
Combined CUDA kernel for KL divergence
This kernel uses vectorized memory accesses with float4 and __ldg for 128-bit aligned load improvements, 
while also performing an efficient shared memory reduction for the partial sums. 
It processes the bulk of the data in vectorized form, with a fallback scalar loop for leftover elements.
*/

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Combined kernel: vectorized loads with __ldg and shared memory reduction
__global__ void kl_div_kernel_combined(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Shared memory for block-wise reduction
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int global_thread_idx = blockIdx.x * blockDim.x + tid;
    int blockStride = blockDim.x * gridDim.x;

    float sum = 0.0f;

    // Process vectorized elements using float4 for 128-bit aligned loads
    int n_vec = n / 4; // number of vectorized groups
    const float4* logp_vec = reinterpret_cast<const float4*>(log_predictions);
    const float4* targ_vec = reinterpret_cast<const float4*>(targets);

    int vec_idx = global_thread_idx;
    while (vec_idx < n_vec) {
        float4 lp = __ldg(&logp_vec[vec_idx]);
        float4 tt = __ldg(&targ_vec[vec_idx]);
        
        sum += expf(lp.x) - tt.x * lp.x;
        sum += expf(lp.y) - tt.y * lp.y;
        sum += expf(lp.z) - tt.z * lp.z;
        sum += expf(lp.w) - tt.w * lp.w;

        vec_idx += blockStride;
    }

    // Process remaining elements with scalar loads
    int remaining_start = n_vec * 4;
    int scalar_idx = remaining_start + global_thread_idx;
    while (scalar_idx < n) {
        float lp = __ldg(&log_predictions[scalar_idx]);
        float tt = __ldg(&targets[scalar_idx]);
        sum += expf(lp) - tt * lp;
        scalar_idx += blockStride;
    }

    // Store partial sum in shared memory
    smem[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    // Block-level reduction result is atomically added to global output
    if (tid == 0) {
        atomicAdd(output, smem[0]);
    }
}

// Host function to launch the combined kernel
// It computes the average KL divergence by normalizing the accumulated result.

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    // Create output tensor (accumulator) with one element
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    // Determine grid size based on vectorized steps (each thread processes 4 elements per iteration)
    const int blocks = (n + threads * 4 - 1) / (threads * 4);
    const int shared_mem = threads * sizeof(float);

    kl_div_kernel_combined<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA Combined Optimized)");
}
