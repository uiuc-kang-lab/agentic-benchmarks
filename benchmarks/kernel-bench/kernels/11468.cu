#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel using 2D grid and block indexing to map a flat 1D input onto a nearly square 2D domain
__global__ void kl_div_kernel_2d_indexing(
    const float* __restrict__ log_predictions,
    const float* __restrict__ targets,
    float* __restrict__ output,
    const int n,
    const int total_width) {

    // Compute global 2D thread coordinates
    int global_x = threadIdx.x + blockIdx.x * blockDim.x;
    int global_y = threadIdx.y + blockIdx.y * blockDim.y;
    // Flatten the 2D coordinate into a 1D index (row-major order)
    int idx = global_y * total_width + global_x;

    // Compute the stride: total number of threads covering one full pass of the data
    int stride = total_width * gridDim.y * blockDim.y;

    float local_sum = 0.0f;
    // Grid-stride loop: Each thread processes multiple elements
    for (; idx < n; idx += stride) {
        float lp = log_predictions[idx];
        float t  = targets[idx];
        local_sum += expf(lp) - t * lp;
    }

    // Flatten the 2D block index into a 1D thread id
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    extern __shared__ float shared[];
    shared[tid] = local_sum;
    __syncthreads();

    // Perform reduction in shared memory (assumes blockDim.x * blockDim.y is a power of two)
    for (int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 of each block atomically adds the block's sum to the global output
    if (tid == 0) {
        atomicAdd(output, shared[0]);
    }
}

// CUDA forward function with 2D indexing optimization
torch::Tensor kl_div_cuda_forward(
    torch::Tensor log_predictions,
    torch::Tensor targets) {

    const int n = log_predictions.numel();
    auto output = torch::zeros({1}, log_predictions.options());

    // Define 2D block dimensions (16x16 = 256 threads per block)
    const int block_x = 16;
    const int block_y = 16;
    dim3 block(block_x, block_y);

    // Map the flat input to a nearly square 2D domain
    int total_columns = ceilf(sqrtf((float)n));
    int total_rows = (n + total_columns - 1) / total_columns;

    // Determine grid dimensions based on block size
    int grid_x = (total_columns + block_x - 1) / block_x;
    int grid_y = (total_rows + block_y - 1) / block_y;
    dim3 grid(grid_x, grid_y);

    int shared_mem_size = block_x * block_y * sizeof(float);
    // total_width is the effective width of the 2D mapping
    int total_width = grid_x * block_x;

    kl_div_kernel_2d_indexing<<<grid, block, shared_mem_size>>>(
        log_predictions.data_ptr<float>(),
        targets.data_ptr<float>(),
        output.data_ptr<float>(),
        n,
        total_width
    );

    return output / static_cast<float>(n);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &kl_div_cuda_forward, "KL divergence forward with 2D indexing optimization (CUDA)");
}
