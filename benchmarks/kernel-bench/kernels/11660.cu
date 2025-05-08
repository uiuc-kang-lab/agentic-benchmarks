#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function for computing KL divergence element
__device__ __forceinline__ float compute_kl_element(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Warp-level reduction using shuffle down
__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Main CUDA kernel with optimized reductions
__global__ void kl_div_kernel_reduction_optimized(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets, 
    float* __restrict__ output,
    const int n) {
    // Shared memory for partial sums
    extern __shared__ float partial_sums[];

    // Get global thread ID
    const unsigned int tid = threadIdx.x;
    const unsigned int global_idx = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;

    // Loop over input, compute local sum and store in shared memory
    float sum = 0.0f;
    for (int idx = global_idx; idx < n; idx += stride) {
        sum += compute_kl_element(log_predictions[idx], targets[idx]);
    }

    // Each thread writes its sum to shared memory
    partial_sums[tid] = sum;
    __syncthreads();

    // Perform block-wide reduction
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            partial_sums[tid] += partial_sums[tid + offset];
        }
        __syncthreads();
    }

    // Perform warp-level reduction and write result
    if (tid < 32) {
        float warp_sum = warp_reduce_sum(partial_sums[tid]);
        if (tid == 0) atomicAdd(output, warp_sum);
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);

    kl_div_kernel_reduction_optimized<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA)");
}