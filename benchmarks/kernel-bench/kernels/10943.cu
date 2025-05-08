#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

// Kernel that utilizes shared memory to tile data for further reduction of global memory latency
// This approach minimizes redundant data fetching from global memory

template <typename scalar_t>
__global__ void shared_memory_optimized_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double s_preds[BLOCK_SIZE];
    __shared__ double s_tgts[BLOCK_SIZE];
    __shared__ double smem[BLOCK_SIZE];
    
    double thread_sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop over the input arrays
    while (idx < num_elements) {
        // Load elements into shared memory
        s_preds[threadIdx.x] = (idx < num_elements) ? static_cast<double>(preds[idx]) : 0.0;
        s_tgts[threadIdx.x] = (idx < num_elements) ? static_cast<double>(tgts[idx]) : 0.0;
        __syncthreads();

        // Calculate squared differences
        for (int j = threadIdx.x; j < BLOCK_SIZE && (blockIdx.x * BLOCK_SIZE + j) < num_elements; j += blockDim.x) {
            double diff = s_preds[j] - s_tgts[j];
            thread_sum += diff * diff;
        }
        __syncthreads();

        idx += stride;
    }

    smem[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            smem[threadIdx.x] += smem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Accumulate each block's contribution atomically
    if (threadIdx.x == 0) {
        atomicAdd(sum_out, smem[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size = std::min(grid_size, 1024);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "shared_memory_optimized_mse_cuda", ([&] {
        shared_memory_optimized_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Shared Memory Optimized MSE forward (CUDA)");
}