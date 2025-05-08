#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel to compute KL divergence with even workload distribution by partitioning the input array across blocks
__global__ void even_workload_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Determine the contiguous chunk this block will process
    int chunk_size = (n + gridDim.x - 1) / gridDim.x;  // ceiling division
    int start = blockIdx.x * chunk_size;
    int end = start + chunk_size;
    if (end > n) end = n;

    float sum = 0.0f;

    // Each thread processes a subrange within the block's chunk
    for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
        float log_val = __ldg(&log_predictions[i]);
        float target_val = __ldg(&targets[i]);
        sum += __expf(log_val) - target_val * log_val;
    }

    // Perform warp-level reduction
    const int warp_size = 32;
    int lane_id = threadIdx.x % warp_size;
    int warp_id = threadIdx.x / warp_size;
    int warps_per_block = blockDim.x / warp_size;

    // Reduce within a warp using shuffle operations
    for (int offset = warp_size/2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Allocate shared memory for warp sums
    extern __shared__ float shared_warp_sums[];
    if (lane_id == 0) {
        shared_warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // First warp reduces the partial sums from all warps
    if (warp_id == 0) {
        float block_sum = (lane_id < warps_per_block) ? shared_warp_sums[lane_id] : 0.0f;
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane_id == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host function to launch the kernel

torch::Tensor even_workload_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Set block size and choose number of blocks to evenly partition the workload
    int block_size = 256;
    // Aim for about block_size*8 elements per block
    int desired_chunk = block_size * 8;
    int blocks = (n + desired_chunk - 1) / desired_chunk;
    if (blocks < 1) blocks = 1;
    if (blocks > 256) blocks = 256;

    // Shared memory: one float per warp
    int num_warps = block_size / 32;
    int shared_mem = num_warps * sizeof(float);

    even_workload_kl_div_kernel<<<blocks, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &even_workload_kl_div_forward, "KLDivLoss with even workload distribution (CUDA)");
}
