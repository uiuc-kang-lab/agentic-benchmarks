#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// New CUDA kernel that combines warp-level reduction (minimal sync) with dynamic block sizing and __ldg loads.
__global__ void efficient_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    // Calculate number of warps per block (assumes blockDim.x is a multiple of warp_size, else rounds up)
    const int num_warps = (blockDim.x + warp_size - 1) / warp_size;

    float sum = 0.0f;
    
    // Grid-stride loop with __ldg for better caching
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        float log_pred = __ldg(&log_predictions[i]);
        float tgt = __ldg(&targets[i]);
        sum += expf(log_pred) - tgt * log_pred;
    }

    // Intra-warp reduction using warp shuffle
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Allocate shared memory for warp-level partial sums
    extern __shared__ float warp_sums[];
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Let the first warp aggregate the results from each warp
    if (warp_id == 0) {
        float block_sum = (lane < num_warps) ? warp_sums[lane] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host function for launching the kernel with dynamic block sizing
torch::Tensor efficient_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Dynamic selection of block size based on problem size
    int best_block_size = 256; // default
    if (n > 65536) {
        best_block_size = 512;
    } else if (n < 8192) {
        best_block_size = 128;
    }

    const int num_warps = (best_block_size + 31) / 32;  // number of warps per block
    const int max_blocks = 256;  // cap for potential blocks
    int blocks = std::min(max_blocks, (n + best_block_size - 1) / best_block_size);
    const int shared_mem = num_warps * sizeof(float);

    efficient_kl_div_kernel<<<blocks, best_block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &efficient_kl_div_forward, "Efficient KL divergence forward (CUDA)");
}
