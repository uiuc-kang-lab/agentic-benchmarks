#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel using shared memory tiling to cache frequently accessed data
__global__ void kl_div_kernel_tiled(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Each thread will process 4 elements per tile
    const int num_per_thread = 4;
    const int tile_size = blockDim.x * num_per_thread;  // Total elements per tile

    // Allocate shared memory: first tile_size for log_predictions, next tile_size for targets
    extern __shared__ float s_data[];
    float* s_log = s_data;            // shared memory for log_predictions
    float* s_target = s_data + tile_size; // shared memory for targets

    float sum = 0.0f;

    // Loop over tiles assigned to this block in a round-robin manner
    // Each block processes tiles: tile_index = blockIdx.x, blockIdx.x + gridDim.x, ...
    for (int tile_index = blockIdx.x; tile_index * tile_size < n; tile_index += gridDim.x) {
        int base_idx = tile_index * tile_size;
        
        // Cooperative loading of a tile from global memory into shared memory
        for (int j = 0; j < num_per_thread; ++j) {
            int global_idx = base_idx + threadIdx.x + j * blockDim.x;
            int smem_idx = threadIdx.x + j * blockDim.x;
            if (global_idx < n) {
                s_log[smem_idx] = log_predictions[global_idx];
                s_target[smem_idx] = targets[global_idx];
            } else {
                s_log[smem_idx] = 0.0f;
                s_target[smem_idx] = 0.0f;
            }
        }
        __syncthreads();
        
        // Process the tile from shared memory
        for (int j = 0; j < num_per_thread; ++j) {
            int smem_idx = threadIdx.x + j * blockDim.x;
            float log_val = s_log[smem_idx];
            float target_val = s_target[smem_idx];
            sum += expf(log_val) - target_val * log_val;
        }
        __syncthreads();
    }

    // Block-wide reduction across threads using shared memory (reusing s_data)
    s_data[threadIdx.x] = sum; 
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_data[threadIdx.x] += s_data[threadIdx.x + stride];
        }
        __syncthreads();
    }
    
    // First thread in the block atomically adds the block sum to the global output
    if (threadIdx.x == 0) {
        atomicAdd(output, s_data[0]);
    }
}


torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    
    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    const int threads = 256;
    const int num_per_thread = 4;
    const int tile_size = threads * num_per_thread; // Number of elements loaded per tile per block
    
    // Compute total number of tiles needed
    int total_tiles = (n + tile_size - 1) / tile_size;
    // Launch one block per tile if possible
    int blocks = min(total_tiles, (n + threads * num_per_thread - 1) / (threads * num_per_thread));
    
    // Shared memory size: 2 arrays of tile_size floats
    int shared_mem_size = 2 * tile_size * sizeof(float);
    
    kl_div_kernel_tiled<<<blocks, threads, shared_mem_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    
    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward using shared memory tiling (CUDA)");
}
