#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function: Compute KL divergence for a single element
__device__ inline float compute_kldiv_value(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// Device function: Perform block-level reduction on shared memory
__device__ inline float block_reduce_sum(float *sdata, int tid, int block_size) {
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    return sdata[0];
}

// CUDA kernel using modular device functions for KL divergence computation
__global__ void kl_div_kernel_modular(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {

    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    for (int idx = gid; idx < n; idx += stride) {
        local_sum += compute_kldiv_value(log_predictions[idx], targets[idx]);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    float block_sum = block_reduce_sum(sdata, tid, blockDim.x);
    if (tid == 0) {
        atomicAdd(output, block_sum);
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward_modular(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);

    kl_div_kernel_modular<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_modular, "Modular KL divergence forward (CUDA)");
}
