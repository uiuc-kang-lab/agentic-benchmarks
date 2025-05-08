#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define tile size for shared memory prefetching
#define TILE_SIZE 1024

// Kernel that leverages shared memory to prefetch tiles of data and reduce global memory latency
__global__ void kl_div_kernel_shared(
    const float * __restrict__ log_predictions,
    const float * __restrict__ targets,
    float * __restrict__ output,
    const int n) {

    // Dynamically allocated shared memory layout:
    // First TILE_SIZE floats for log_predictions tile (s_log)
    // Next TILE_SIZE floats for targets tile (s_targ)
    // Last (blockDim.x / 32) floats for warp-level reduction scratch (s_warp)
    extern __shared__ float shared_mem[];
    float *s_log  = shared_mem;                     // indices [0, TILE_SIZE)
    float *s_targ = s_log + TILE_SIZE;               // indices [TILE_SIZE, 2*TILE_SIZE)
    int warp_count = blockDim.x / 32;
    float *s_warp = s_targ + TILE_SIZE;              // indices [2*TILE_SIZE, 2*TILE_SIZE + warp_count)

    float local_sum = 0.0f;

    // Grid-stride loop over tiles of size TILE_SIZE
    // Each block starts at an offset based on its blockIdx.x and then strides by gridDim.x*TILE_SIZE
    for (int tile_start = blockIdx.x * TILE_SIZE; tile_start < n; tile_start += gridDim.x * TILE_SIZE) {
        // Load a tile of data from global memory to shared memory
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            int idx = tile_start + i;
            if (idx < n) {
                s_log[i]  = log_predictions[idx];
                s_targ[i] = targets[idx];
            } else {
                s_log[i]  = 0.0f;
                s_targ[i] = 0.0f;
            }
        }
        __syncthreads();  // Ensure the tile is loaded

        // Process the tile from shared memory
        for (int i = threadIdx.x; i < TILE_SIZE; i += blockDim.x) {
            int idx = tile_start + i;
            if (idx < n) {
                float lp = s_log[i];
                float tt = s_targ[i];
                local_sum += expf(lp) - tt * lp;
            }
        }
        __syncthreads();  // Prepare for the next tile load
    }

    // Reduce local_sum across the block using a two-level (warp-level then block-level) reduction
    float sum_val = local_sum;
    // Intra-warp reduction
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        sum_val += __shfl_down_sync(0xffffffff, sum_val, offset);
    }
    
    // Write each warp's result to shared memory
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        int warp_id = threadIdx.x >> 5;  // division by 32
        s_warp[warp_id] = sum_val;
    }
    __syncthreads();

    float block_sum = 0.0f;
    // Let first warp load the per-warp sums
    if (threadIdx.x < warp_count) {
        block_sum = s_warp[threadIdx.x];
    }
    // Final reduction within the first warp
    if (threadIdx.x < 32) {
        for (int offset = warpSize/2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(0xffffffff, block_sum, offset);
        }
    }

    // One thread per block adds the block's result to the global output
    if (threadIdx.x == 0) {
        atomicAdd(output, block_sum);
    }
}

// CUDA forward function exposed to PyTorch
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());
    
    const int threads = 256;  // number of threads per block
    int blocks = 1024;        // number of blocks; adjust based on input size and GPU occupancy

    int warp_count = threads / 32;
    // Total shared memory size in bytes: 2*TILE_SIZE floats for the tile + warp_count floats for reduction
    size_t shmem_size = (2 * TILE_SIZE + warp_count) * sizeof(float);

    kl_div_kernel_shared<<<blocks, threads, shmem_size>>>(
         log_predictions.data_ptr<float>(),
         targets.data_ptr<float>(),
         output.data_ptr<float>(),
         n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with shared memory (CUDA)");
}
