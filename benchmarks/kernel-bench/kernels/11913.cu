#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32

// Warp-level reduction using shuffle down intrinsic
__device__ __forceinline__ float warp_reduce(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Kernel employing 2D grid and 2D block indexing for efficient mapping
__global__ void kl_div_kernel_indexing_2d(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n) {

    // Compute 2D block and grid indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Block dimensions
    int blockDimX = blockDim.x;  // e.g., 16
    int blockDimY = blockDim.y;  // e.g., 16
    
    // Grid dimensions (using only gridDim.x for linearization in x-direction)
    int gridDimX = gridDim.x;

    // Compute a unique global thread index by linearizing 2D block indices:
    int global_index = tx + bx * blockDimX + (ty + by * blockDimY) * (gridDimX * blockDimX);
    
    // Total number of threads launched
    int total_threads = gridDim.x * blockDim.x * gridDim.y * blockDim.y;
    
    float local_sum = 0.0f;

    // Grid-stride loop: each thread processes multiple elements
    for (int idx = global_index; idx < n; idx += total_threads) {
        float log_pred = log_predictions[idx];
        float target = targets[idx];
        local_sum += expf(log_pred) - target * log_pred;
    }
    
    // Warp-level reduction within each thread's subgroup
    local_sum = warp_reduce(local_sum);

    // Allocate shared memory for block-level reduction
    extern __shared__ float shared_data[];
    int linear_tid = threadIdx.x + threadIdx.y * blockDim.x;  // Linear thread id within block

    // Each warp's leader writes its reduced sum to shared memory
    if ((linear_tid % WARP_SIZE) == 0) {
        shared_data[linear_tid / WARP_SIZE] = local_sum;
    }
    __syncthreads();

    // Final reduction: first warp in the block combines results
    int numWarps = (blockDim.x * blockDim.y) / WARP_SIZE;
    if (linear_tid < WARP_SIZE) {
        float warp_sum = (linear_tid < numWarps) ? shared_data[linear_tid] : 0.0f;
        warp_sum = warp_reduce(warp_sum);
        if (linear_tid == 0) {
            atomicAdd(output, warp_sum);
        }
    }
}

// Host function called from PyTorch
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {
    int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Define 2D block dimensions, e.g., 16x16 (=256 threads per block)
    dim3 block(16, 16);
    int threads_per_block = block.x * block.y;  // 256

    // Determine grid dimensions based on total work
    // Here we fix grid.x and compute grid.y to ensure enough threads cover all n elements
    int total_blocks = (n + threads_per_block - 1) / threads_per_block;
    int grid_x = 32;  // fixed dimension for grid x
    int grid_y = (total_blocks + grid_x - 1) / grid_x;
    dim3 grid(grid_x, grid_y);

    // Shared memory required: one float per warp per block
    int shared_mem = (threads_per_block / WARP_SIZE) * sizeof(float);

    kl_div_kernel_indexing_2d<<<grid, block, shared_mem>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward (CUDA 2D indexing)");
}
