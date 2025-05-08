#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kldiv_shared_memory_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    extern __shared__ float s_buffer[];
    float* s_logs = s_buffer;
    float* s_targets = &s_buffer[blockDim.x];
    
    unsigned int tid = threadIdx.x;
    unsigned int bid = blockIdx.x;
    unsigned int bdim = blockDim.x;
    float sum = 0.0f;

    // Process data in tiles using shared memory
    for (unsigned int tile_base = bid * bdim; tile_base < n; tile_base += gridDim.x * bdim) {
        unsigned int load_idx = tile_base + tid;
        
        // Cooperative loading into shared memory
        if (load_idx < n) {
            s_logs[tid] = log_predictions[load_idx];
            s_targets[tid] = targets[load_idx];
        } else {
            s_logs[tid] = 0.0f;
            s_targets[tid] = 0.0f;
        }
        __syncthreads();

        // Process current tile from shared memory
        unsigned int compute_idx = tile_base + tid;
        if (compute_idx < n) {
            sum += expf(s_logs[tid]) - s_targets[tid] * s_logs[tid];
        }
        __syncthreads();
    }

    // Warp-level reduction
    unsigned int mask = 0xffffffff;
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Block-level reduction
    __shared__ float block_sum[32];
    if (tid % 32 == 0) {
        block_sum[tid/32] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        float val = (tid < (bdim + 31)/32) ? block_sum[tid] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (tid == 0) {
            atomicAdd(output, val);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    const torch::Tensor& log_predictions,
    const torch::Tensor& targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const unsigned int threads = 512;
    const unsigned int blocks = (n + threads - 1) / threads;
    const size_t shared_mem = 2 * threads * sizeof(float);

    kldiv_shared_memory_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence with shared memory optimization");
}