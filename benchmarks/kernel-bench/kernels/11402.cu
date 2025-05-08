#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

// CUDA kernel for KL divergence calculation using 2D grid indexing
__global__ void kl_div_kernel(
    const float* log_predictions,
    const float* targets,
    float* output,
    int n) {

    // Compute global thread index using 2D grid indexing
    int global_block_idx = blockIdx.y * gridDim.x + blockIdx.x;
    int global_thread_idx = global_block_idx * blockDim.x + threadIdx.x;
    int total_threads = gridDim.x * gridDim.y * blockDim.x;

    extern __shared__ float shared_sum[];
    float sum = 0.0f;

    // Grid-stride loop to process multiple elements per thread
    for (int i = global_thread_idx; i < n; i += total_threads) {
        float log_val = log_predictions[i];
        float target = targets[i];
        sum += expf(log_val) - target * log_val;
    }

    shared_sum[threadIdx.x] = sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Write the block result to global memory
    if (threadIdx.x == 0) {
        atomicAdd(output, shared_sum[0]);
    }
}


torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Define thread and block configuration
    int threads = 256;
    int total_blocks = (n + threads - 1) / threads;
    // Use a 2D grid: gridDim.x is capped to 1024, gridDim.y covers remaining blocks
    int grid_x = total_blocks > 1024 ? 1024 : total_blocks;
    int grid_y = (total_blocks + grid_x - 1) / grid_x;

    dim3 blocks(grid_x, grid_y);
    dim3 threadsPerBlock(threads);
    int shared_mem = threads * sizeof(float);

    kl_div_kernel<<<blocks, threadsPerBlock, shared_mem>>>(
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
