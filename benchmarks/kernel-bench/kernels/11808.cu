#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function: Compute KL divergence for a single element
__device__ inline float compute_kldiv_value(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function: Perform block-level reduction on shared memory
__device__ inline float block_reduce_sum(float *sdata, int tid) {
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    return sdata[0];
}

// CUDA kernel: Optimized using device functions and improved memory access patterns
__global__ void optimized_kl_div_kernel(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    // Reorder accesses to boost memory coalescing
    float local_sum = 0.0f;
    for (int idx = gid; idx < n; idx += stride) {
        local_sum += compute_kldiv_value(log_predictions[idx], targets[idx]);
    }

    // Store in shared memory for block-level reduction
    sdata[tid] = local_sum;
    __syncthreads();

    // Perform block reduction
    float block_sum = block_reduce_sum(sdata, tid);
    if (tid == 0) {
        atomicAdd(output, block_sum);
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward_optimized(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    // Automatically determine the appropriate number of blocks based on input size
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);

    optimized_kl_div_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_optimized, "Optimized KL divergence forward (CUDA)");
}