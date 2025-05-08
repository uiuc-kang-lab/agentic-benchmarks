#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

// This kernel leverages shared memory to preload tiles of data from global memory.
// It divides the input into tiles of blockDim.x elements, loading both log_predictions and targets
// into shared memory. Each thread computes its contribution to the KL divergence for its element in the tile.
// After processing all tiles in a grid-stride loop, an intra-warp reduction (using shuffle operations) is performed,
// followed by a block-level reduction using shared memory. The final result is atomically added to the global output.

__global__ void shared_tile_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Allocate shared memory dynamically:
    // - First blockDim.x floats for a tile of log_predictions
    // - Next blockDim.x floats for a tile of targets
    // - Then (blockDim.x / WARP_SIZE) floats for warp-level reduction
    extern __shared__ float sdata[];
    float* log_tile    = sdata;                     // size: blockDim.x floats
    float* target_tile = sdata + blockDim.x;          // size: blockDim.x floats
    float* warp_cache  = sdata + 2 * blockDim.x;       // size: blockDim.x / WARP_SIZE floats

    float local_sum = 0.0f;
    int total_tiles = (n + blockDim.x - 1) / blockDim.x;

    // Loop over tiles in a grid-stride manner
    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < n) {
            log_tile[threadIdx.x] = log_predictions[idx];
            target_tile[threadIdx.x] = targets[idx];
        } else {
            // Fill with zero to avoid garbage values
            log_tile[threadIdx.x] = 0.0f;
            target_tile[threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute contribution from this tile
        float val = expf(log_tile[threadIdx.x]) - target_tile[threadIdx.x] * log_tile[threadIdx.x];
        local_sum += val;
        __syncthreads();
    }

    // Intra-warp reduction using shuffle operations
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Each warp's first thread stores its reduced sum into warp_cache
    if ((threadIdx.x % WARP_SIZE) == 0) {
        warp_cache[threadIdx.x / WARP_SIZE] = local_sum;
    }
    __syncthreads();

    // Final reduction across warps, performed by the first warp
    int warp_count = blockDim.x / WARP_SIZE;
    if (threadIdx.x < warp_count) {
        local_sum = warp_cache[threadIdx.x];
        for (int offset = warp_count / 2; offset > 0; offset /= 2) {
            local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, local_sum);
        }
    }
}

// Host function to launch the shared-tile KL divergence kernel
torch::Tensor shared_tile_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    int blocks = (n + threads - 1) / threads;
    // Limit the maximum blocks to ensure sufficient workload per block
    blocks = min(blocks, 256);

    // Calculate shared memory size:
    //  - log_tile: blockDim.x floats
    //  - target_tile: blockDim.x floats
    //  - warp_cache: (blockDim.x / WARP_SIZE) floats
    int warp_count = threads / WARP_SIZE;
    size_t shared_mem = (2 * threads + warp_count) * sizeof(float);

    shared_tile_kl_kernel<<<blocks, threads, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_tile_kl_forward, "KL divergence with shared memory tiling (CUDA)");
}
