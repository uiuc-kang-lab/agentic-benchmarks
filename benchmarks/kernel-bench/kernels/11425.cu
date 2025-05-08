#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Warp-level reduction using shfl_down_sync for improved efficiency
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Combined kernel using grid-stride loop and warp-level reduction with shared memory
__global__ void combined_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int warp_size = 32;
    const int lane = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;
    const int grid_stride = gridDim.x * blockDim.x;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    // Process multiple elements per thread using grid-stride loop
    while (idx < n) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        sum += expf(log_pred) - target * log_pred;
        idx += grid_stride;
    }

    // Each warp performs an efficient intra-warp reduction
    sum = warp_reduce_sum(sum);

    // Allocate shared memory for per-warp partial sums
    extern __shared__ float warp_sums[];
    if (lane == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // First warp finalizes the block-level sum
    if (warp_id == 0) {
        sum = (lane < warps_per_block) ? warp_sums[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (lane == 0) {
            atomicAdd(output, sum);
        }
    }
}

// Host function to launch the CUDA kernel
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    
    // Optimal configuration: 256 threads per block and up to 256 blocks
    const int threads = 256;
    const int max_blocks = 256;
    const int blocks = min(max_blocks, (n + threads - 1) / threads);

    // Shared memory: one float per warp
    const int warps_per_block = threads / 32;
    const int shared_mem = warps_per_block * sizeof(float);

    auto output = torch::zeros({1}, log_predictions.options());

    combined_kl_div_kernel<<<blocks, threads, shared_mem>>>(
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
