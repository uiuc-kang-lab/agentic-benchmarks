#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define warp size constant
constexpr int WARP_SIZE = 32;

// Kernel: Each thread performs a grid-stride loop to accumulate a local sum,
// then reduces within a warp using __shfl_down_sync(), writes warp results to shared memory,
// and finally the first warp reduces these shared values to produce the block sum, which
// is atomically added to the global output.
__global__ void shared_warp_reduced_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * blockDim.x;

    // Each thread computes a local sum over its assigned elements using a grid-stride loop
    float local_sum = 0.0f;
    for (int idx = tid; idx < n; idx += total_threads) {
        float lp = log_predictions[idx];
        float t = targets[idx];
        local_sum += expf(lp) - t * lp;
    }

    // Intra-warp reduction using shuffle operations
    unsigned int mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Each warp's lane0 writes its result to shared memory
    extern __shared__ float shared_data[];  // size: (blockDim.x / WARP_SIZE) floats
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        shared_data[warp_id] = local_sum;
    }
    __syncthreads();

    // Final reduction by the first warp in the block
    float block_sum = 0.0f;
    int num_warps = blockDim.x / WARP_SIZE;
    if (threadIdx.x < num_warps) {
        block_sum = shared_data[lane];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
        if (lane == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host function that launches the CUDA kernel
torch::Tensor shared_warp_reduced_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    // Limit the number of blocks to ensure sufficient work per block
    blocks = (blocks < 256) ? blocks : 256;
    
    // Shared memory size: one float per warp
    int shared_mem = (threads / WARP_SIZE) * sizeof(float);
    
    shared_warp_reduced_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n);

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_warp_reduced_kl_forward, "KL divergence optimized with shared memory and warp-level reduction (CUDA)");
}
