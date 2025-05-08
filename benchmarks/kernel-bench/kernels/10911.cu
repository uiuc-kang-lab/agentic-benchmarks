#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_DIM_X = 16;
static const int BLOCK_DIM_Y = 16;

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_DIM_X * BLOCK_DIM_Y];
    
    // 2D thread indexing
    const int tid = threadIdx.y * BLOCK_DIM_X + threadIdx.x;
    const int num_threads_per_block = BLOCK_DIM_X * BLOCK_DIM_Y;
    const int bid = blockIdx.y * gridDim.x + blockIdx.x;
    const int num_blocks = gridDim.x * gridDim.y;
    
    // Calculate starting index for this thread
    int idx = bid * num_threads_per_block + tid;
    const int grid_stride = num_threads_per_block * num_blocks;
    
    double thread_sum = 0.0;
    
    // Grid-stride loop
    while (idx < num_elements) {
        if (idx < num_elements) {
            double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
            thread_sum += diff * diff;
        }
        idx += grid_stride;
    }
    
    // Store in shared memory
    shm[tid] = thread_sum;
    __syncthreads();
    
    // Reduction within shared memory
    for (int s = num_threads_per_block/2; s > 0; s >>= 1) {
        if (tid < s) {
            shm[tid] += shm[tid + s];
        }
        __syncthreads();
    }
    
    // Write result
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
    dim3 block_dim(BLOCK_DIM_X, BLOCK_DIM_Y);
    int grid_x = std::min(32, (int)ceil(sqrt(num_elements / (BLOCK_DIM_X * BLOCK_DIM_Y))));
    int grid_y = grid_x;
    dim3 grid_dim(grid_x, grid_y);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel<scalar_t><<<grid_dim, block_dim>>>(
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