#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// Use a 2D block configuration: 16x16 threads
static const int BLOCK_DIM_X = 16;
static const int BLOCK_DIM_Y = 16;

// CUDA kernel that uses 2D thread and block indexing to compute MSE with grid-stride loop
// The global thread index is computed from 2D block and thread indices

template <typename scalar_t>
__global__ void indexed2d_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    // Compute the number of threads per block (assumed to be BLOCK_DIM_X * BLOCK_DIM_Y)
    const int threads_per_block = blockDim.x * blockDim.y;
    // Compute the thread's linear index within the block
    int local_tid = threadIdx.y * blockDim.x + threadIdx.x;

    // Compute a unique block index from 2D grid
    int global_block_id = blockIdx.y * gridDim.x + blockIdx.x;

    // Compute the global thread index
    int global_tid = global_block_id * threads_per_block + local_tid;

    // Compute grid stride: total number of threads in the grid
    int grid_stride = gridDim.x * gridDim.y * threads_per_block;

    double thread_sum = 0.0;
    
    // Grid-stride loop
    for (int idx = global_tid; idx < num_elements; idx += grid_stride) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
    }

    // Reduction within block using shared memory
    extern __shared__ double shmem[]; // size should be allocated as threads_per_block * sizeof(double)
    shmem[local_tid] = thread_sum;
    __syncthreads();

    // Standard iterative reduction in shared memory
    for (int offset = threads_per_block / 2; offset > 0; offset >>= 1) {
        if (local_tid < offset) {
            shmem[local_tid] += shmem[local_tid + offset];
        }
        __syncthreads();
    }

    // The first thread in the block adds the block's sum to the global sum
    if (local_tid == 0) {
        atomicAdd(sum_out, shmem[0]);
    }
}

// Forward function exposed to Python

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    
    // Create accumulator tensor using double precision
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Define block dimensions
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    const int threads_per_block = BLOCK_DIM_X * BLOCK_DIM_Y;
    
    // Calculate total number of blocks needed
    int totalBlocks = (num_elements + threads_per_block - 1) / threads_per_block;
    // Use a 2D grid: choose grid.x as ceil(sqrt(totalBlocks)) and grid.y accordingly
    int grid_x = static_cast<int>(std::ceil(std::sqrt(static_cast<double>(totalBlocks))));
    int grid_y = (totalBlocks + grid_x - 1) / grid_x;
    dim3 grid_dim(grid_x, grid_y);

    // Launch the kernel with dynamically allocated shared memory for reduction
    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "indexed2d_mse_cuda", ([&] {
        indexed2d_mse_kernel<scalar_t><<<grid_dim, block_dim, threads_per_block * sizeof(double)>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Final MSE: mean value obtained by dividing the total sum by num_elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward with 2D indexing (CUDA)");
}
