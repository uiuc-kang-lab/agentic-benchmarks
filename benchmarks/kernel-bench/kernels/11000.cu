#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Using 16x16 thread blocks for better occupancy
#define BLOCK_DIM_X 16
#define BLOCK_DIM_Y 16

template <typename scalar_t>
__global__ void mse_forward_kernel_2d(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    const int tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
    const int block_size = BLOCK_DIM_X * BLOCK_DIM_Y;
    const int bid = blockIdx.y * gridDim.x + blockIdx.x;
    const int global_idx_base = bid * block_size + tid;
    const int grid_size = gridDim.x * gridDim.y * block_size;
    
    // Shared memory for block reduction
    __shared__ double shm[BLOCK_DIM_X * BLOCK_DIM_Y];
    
    double thread_sum = 0.0;
    
    // Grid-stride loop using 2D indexing
    int idx = global_idx_base;
    while (idx < num_elements) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
        idx += grid_size;
    }
    
    // Store in shared memory
    shm[tid] = thread_sum;
    __syncthreads();
    
    // Reduction within block using 2D thread structure
    for (int stride = block_size/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shm[tid] += shm[tid + stride];
        }
        __syncthreads();
    }
    
    // First thread in block writes result
    if (tid == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Calculate 2D grid dimensions
    const int total_blocks = (num_elements + (BLOCK_DIM_X * BLOCK_DIM_Y) - 1) / (BLOCK_DIM_X * BLOCK_DIM_Y);
    const int grid_dim = static_cast<int>(sqrt(total_blocks)) + 1;
    
    dim3 blocks(grid_dim, grid_dim);
    dim3 threads(BLOCK_DIM_X, BLOCK_DIM_Y);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel_2d<scalar_t><<<blocks, threads>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA)");
}