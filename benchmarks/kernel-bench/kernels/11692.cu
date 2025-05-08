#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Kernel that uses a grid-stride loop, warp-level reduction, and dynamic block size optimization
__global__ void optimal_kl_div_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    const int warp_size = 32;
    const int lane_id = threadIdx.x % warp_size;
    const int warp_id = threadIdx.x / warp_size;
    const int warps_per_block = blockDim.x / warp_size;

    extern __shared__ float shared_data[];  // One float per warp

    float sum = 0.0f;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop over all elements
    for (int i = idx; i < n; i += stride) {
        float log_pred = __ldg(&log_predictions[i]);
        float target = __ldg(&targets[i]);
        sum += expf(log_pred) - target * log_pred;
    }

    // Intra-warp reduction using shuffle operations
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Each warp writes its result to shared memory
    if (lane_id == 0) {
        shared_data[warp_id] = sum;
    }

    __syncthreads(); // Ensure all warps have written their values

    // First warp reduces the partial sums from all warps
    if (warp_id == 0) {
        float block_sum = (lane_id < warps_per_block) ? shared_data[lane_id] : 0.0f;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (lane_id == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host function that selects the optimal block size based on input size and launches the kernel
torch::Tensor optimal_blocksize_kl_div_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Experiment with block sizes (32, 64, 128, 256, 512) based on the input size
    int block_size;
    if (n < 4096) {
        block_size = 32;
    } else if (n < 8192) {
        block_size = 64;
    } else if (n < 32768) {
        block_size = 128;
    } else if (n < 131072) {
        block_size = 256;
    } else {
        block_size = 512;
    }

    // Limit the number of blocks to avoid oversubscription
    const int max_blocks = 256;
    int blocks = min(max_blocks, (n + block_size - 1) / block_size);
    
    // Allocate shared memory: one float per warp
    int num_warps = block_size / 32;
    int shared_mem = num_warps * sizeof(float);

    optimal_kl_div_kernel<<<blocks, block_size, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &optimal_blocksize_kl_div_forward, "KLDivLoss with optimal block size selection (CUDA)");
}
