// Includes
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp reduction using shuffle intrinsics
__inline__ __device__ float warp_reduce_sum(float val) {
    // Unrolling warp reduction
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Optimized KL divergence kernel using grid-stride loop and two-level reduction
__global__ void kl_div_optimized_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Calculate global thread index and grid stride
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_stride = gridDim.x * blockDim.x;

    // Each thread accumulates its partial result
    float sum = 0.0f;
    for (int i = idx; i < n; i += grid_stride) {
        // Use __ldg to load from read-only cache
        float lp = __ldg(&log_predictions[i]);
        float t = __ldg(&targets[i]);
        sum += expf(lp) - t * lp;
    }

    // Intra-warp reduction
    sum = warp_reduce_sum(sum);

    // Allocate shared memory for warp-level partial sums
    extern __shared__ float shared_mem[];  // size = (blockDim.x / warpSize) * sizeof(float)
    int lane = threadIdx.x % warpSize;
    int warp_id = threadIdx.x / warpSize;

    // Write reduced sum of each warp to shared memory
    if (lane == 0) {
        shared_mem[warp_id] = sum;
    }
    __syncthreads();

    // Let the first warp perform the final reduction
    if (warp_id == 0) {
        // Number of warps in this block
        int num_warps = blockDim.x / warpSize;
        // Each thread in the first warp loads one warp's sum
        float block_sum = (lane < num_warps) ? shared_mem[lane] : 0.0f;
        block_sum = warp_reduce_sum(block_sum);
        if (lane == 0) {
            // Atomic addition of the block's sum to global output
            atomicAdd(output, block_sum);
        }
    }
}

// Host function called from PyTorch
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Configure threads and blocks
    const int threads = 256;
    const int blocks = min(256, (n + threads - 1) / threads);
    // Shared memory size: one float per warp in the block
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);

    // Launch kernel
    kl_div_optimized_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    // Normalize the output by the number of elements
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "Optimized KL divergence forward (CUDA)");
}
