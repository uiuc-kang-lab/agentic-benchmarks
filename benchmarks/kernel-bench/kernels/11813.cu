#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Device function to compute KL divergence for a single element
__device__ inline float compute_kldiv_value(float log_pred, float target) {
    return expf(log_pred) - target * log_pred;
}

// CUDA kernel with manual loop unrolling for grid-stride loop and reduction
__global__ void kl_div_kernel_unrolled(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float local_sum = 0.0f;
    int idx = gid;

    // Unrolled grid-stride loop: process 4 elements per iteration
    #pragma unroll 4
    for (; idx + 3 * stride < n; idx += stride * 4) {
        local_sum += compute_kldiv_value(log_predictions[idx], targets[idx])
                   + compute_kldiv_value(log_predictions[idx + stride], targets[idx + stride])
                   + compute_kldiv_value(log_predictions[idx + 2 * stride], targets[idx + 2 * stride])
                   + compute_kldiv_value(log_predictions[idx + 3 * stride], targets[idx + 3 * stride]);
    }

    // Process any remaining elements
    for (; idx < n; idx += stride) {
        local_sum += compute_kldiv_value(log_predictions[idx], targets[idx]);
    }

    // Allocate shared memory for block-level reduction
    extern __shared__ float shared_sum[];
    shared_sum[tid] = local_sum;
    __syncthreads();

    // Manually unrolled reduction in shared memory
    if (blockDim.x >= 512) {
        if (tid < 256) { shared_sum[tid] += shared_sum[tid + 256]; }
        __syncthreads();
    }
    if (blockDim.x >= 256) {
        if (tid < 128) { shared_sum[tid] += shared_sum[tid + 128]; }
        __syncthreads();
    }
    if (blockDim.x >= 128) {
        if (tid < 64) { shared_sum[tid] += shared_sum[tid + 64]; }
        __syncthreads();
    }

    // Unroll the final warp while relying on warp-synchronous programming
    if (tid < 32) {
        volatile float *vshared = shared_sum;
        vshared[tid] += vshared[tid + 32];
        vshared[tid] += vshared[tid + 16];
        vshared[tid] += vshared[tid + 8];
        vshared[tid] += vshared[tid + 4];
        vshared[tid] += vshared[tid + 2];
        vshared[tid] += vshared[tid + 1];
    }

    // Write block's reduced sum to global output using atomic add
    if (tid == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}

// CUDA forward function
torch::Tensor kl_div_cuda_forward_unrolled(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int blocks = min((n + threads - 1) / threads, 1024);
    const int shared_mem = threads * sizeof(float);

    kl_div_kernel_unrolled<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward_unrolled, "KL divergence forward with unrolled loops (CUDA)");
}
