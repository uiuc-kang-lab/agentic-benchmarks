#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Kernel that uses a grid-stride loop to handle workloads larger than available threads
__global__ void kl_div_stride_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    float sum = 0.0f;
    // Grid-stride loop: each thread processes multiple elements
    for (int idx = global_idx; idx < n; idx += stride) {
        float lp = log_predictions[idx];
        float t  = targets[idx];
        sum += expf(lp) - t * lp;
    }

    // Warp-level reduction using shuffle
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Each warp's lane 0 writes its result to shared memory
    __shared__ float warp_sums[32]; // Supports up to 32 warps per block
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces the per-warp partial sums
    if (warp_id == 0) {
        float block_sum = (tid < (blockDim.x / WARP_SIZE)) ? warp_sums[lane] : 0.0f;
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0)
            atomicAdd(output, block_sum);
    }
}

// CUDA forward function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;
    // Compute number of blocks needed to cover all elements
    const int blocks = (n + threads - 1) / threads;
    
    kl_div_stride_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA Stride Loop with Boundary Check)");
}
