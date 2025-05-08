#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets,
    float* output,
    const int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    extern __shared__ float warp_sums[];
    const int warps_per_block = blockDim.x / 32;
    float thread_sum = 0.0f;

    // Grid-stride loop with uniform iterations per warp
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        thread_sum += expf(log_pred) - target * log_pred;
        idx += blockDim.x * gridDim.x;
    }

    // Warp-level reduction using shuffle
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }

    // Store warp sums in shared memory
    if (threadIdx.x % 32 == 0) {
        warp_sums[threadIdx.x / 32] = thread_sum;
    }
    __syncthreads();

    // Block-level reduction of warp sums
    if (threadIdx.x < warps_per_block) {
        float block_sum = warp_sums[threadIdx.x];
        for (int stride = warps_per_block / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                warp_sums[threadIdx.x] += warp_sums[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, warp_sums[0]);
        }
    }
}

torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem = (threads / 32) * sizeof(float);
    
    kl_div_kernel<<<blocks, threads, shared_mem>>>(
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