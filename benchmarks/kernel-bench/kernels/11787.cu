#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define constants
constexpr int WARP_SIZE = 32;
constexpr int ELEMENTS_PER_THREAD = 4;  // Each block will process a tile of (blockDim.x * ELEMENTS_PER_THREAD) elements

// Kernel: Uses shared memory to cache tiles of log_predictions and targets to reduce global memory latency
// and then computes the KL divergence over the tile. A two-stage reduction (warp shuffle and block-wide shared memory reduction)
// ensures proper accumulation without race conditions.
__global__ void shared_tiled_kl_kernel(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {
    
    // Determine the tile size for this block
    const int tile_size = blockDim.x * ELEMENTS_PER_THREAD;

    // Allocate dynamic shared memory for the tile data.
    // The first tile_size floats will hold log_predictions values,
    // and the next tile_size floats will hold targets values.
    extern __shared__ float shared_mem[];
    float* s_log = shared_mem;                // [0, tile_size)
    float* s_target = shared_mem + tile_size;   // [tile_size, 2*tile_size)

    float thread_sum = 0.0f;

    // Process the input in tiles. Each block processes several tiles in a grid-stride loop over the tile range.
    // The global index offset for each tile is computed using a stride of (gridDim.x * tile_size).
    for (int base = blockIdx.x * tile_size; base < n; base += gridDim.x * tile_size) {
        // Load a tile of data from global memory into shared memory
        // Each thread cooperatively loads ELEMENTS_PER_THREAD (spaced by blockDim.x) elements into the tile
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            int index = base + i;
            if (index < n) {
                s_log[i]    = log_predictions[index];
                s_target[i] = targets[index];
            } else {
                s_log[i] = 0.0f;
                s_target[i] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute KL divergence on the tile stored in shared memory.
        // Each thread processes a subset of the tile by striding over the tile
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            int global_idx = base + i;
            if (global_idx < n) {
                float lp = s_log[i];
                float t  = s_target[i];
                thread_sum += expf(lp) - t * lp;
            }
        }
        __syncthreads();  // Ensure tile memory is not overwritten before complete processing
    }

    // Intra-block reduction of thread_sum
    // First, perform warp-level reduction using shuffle operations
    float sum = thread_sum;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }

    // Use static shared memory for storing the reduced sum from each warp
    __shared__ float warp_sums[32];  // Assuming blockDim.x <= 1024, which is 32 warps max
    int warp_id = threadIdx.x / WARP_SIZE;
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
        warp_sums[warp_id] = sum;
    }
    __syncthreads();

    // Final reduction: Let the first warp reduce the values in warp_sums
    if (threadIdx.x < (blockDim.x / WARP_SIZE)) {
        float block_sum = warp_sums[threadIdx.x];
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(output, block_sum);
        }
    }
}

// Host function to launch the kernel
torch::Tensor shared_tiled_kl_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    // Kernel launch configuration
    const int threads = 256;
    const int tile_size = threads * ELEMENTS_PER_THREAD;  // Number of elements processed per block per tile
    int num_tiles = (n + tile_size - 1) / tile_size;
    // Use a grid with enough blocks to cover the tiles, but cap it to maintain occupancy
    const int blocks = min(num_tiles, 256);
    
    // Dynamic shared memory required: two float arrays of size tile_size each
    size_t shared_mem_bytes = 2 * tile_size * sizeof(float);
    
    shared_tiled_kl_kernel<<<blocks, threads, shared_mem_bytes>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &shared_tiled_kl_forward, "Shared tiled KL divergence (CUDA)");
}
