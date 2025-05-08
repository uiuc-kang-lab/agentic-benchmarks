#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel with manually unrolled reduction loops for improved performance
__global__ void kldiv_unrolled_atomics_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Grid-stride loop: Each thread accumulates its partial sum
    while (idx < n) {
        float log_val = log_predictions[idx];
        float target_val = targets[idx];
        // Compute KL divergence term: exp(log_val) - target_val * log_val
        sum += expf(log_val) - target_val * log_val;
        idx += blockDim.x * gridDim.x;
    }

    // Warp-level reduction using shuffle with loop unrolling
    unsigned int mask = 0xffffffff;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(mask, sum, offset);
    }

    // Store the result of each warp in shared memory
    __shared__ float warpSums[32]; // supports blocks up to 1024 threads (32 warps)
    int warpId = threadIdx.x >> 5;  // Divide by warp size
    if ((threadIdx.x & 31) == 0) {
        warpSums[warpId] = sum;
    }
    __syncthreads();

    // Final reduction across warps, performed by the first warp
    if (threadIdx.x < 32) {
        int numWarps = (blockDim.x + 31) >> 5;
        float val = (threadIdx.x < numWarps) ? warpSums[threadIdx.x] : 0.0f;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(mask, val, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, val);
        }
    }
}

// CUDA function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(const torch::Tensor& log_predictions,
                                    const torch::Tensor& targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;

    kldiv_unrolled_atomics_kernel<<<blocks, threads>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with loop unrolling (CUDA)");
}
