#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

// Kernel that uses shared memory to cache data and reduce global memory latency

template <typename scalar_t>
__global__ void shared_memory_optimized_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    double thread_sum = 0.0;
    const int grid_stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + tid;

    // Load data into shared memory in a coalesced manner
    while (idx < num_elements) {
        double pred_val = static_cast<double>(preds[idx]);
        double tgt_val = static_cast<double>(tgts[idx]);
        double diff = pred_val - tgt_val;
        thread_sum += diff * diff;
        idx += grid_stride;
    }

    smem[tid] = thread_sum;
    __syncthreads();

    // Reduction within shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            smem[tid] += smem[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
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
