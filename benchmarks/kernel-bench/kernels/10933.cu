#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

// CUDA kernel using a grid-stride loop with correct boundary handling
template <typename scalar_t>
__global__ void stride_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    // Shared memory for reduction
    __shared__ double shmem[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    double local_sum = 0.0;

    // Grid-stride loop to handle workloads larger than available threads
    for (; idx < num_elements; idx += stride) {
        // Verify boundary before reading
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        local_sum += diff * diff;
    }

    // Store local sum to shared memory
    shmem[threadIdx.x] = local_sum;
    __syncthreads();

    // Intra-block reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shmem[threadIdx.x] += shmem[threadIdx.x + s];
        }
        __syncthreads();
    }

    // The first thread of each block updates the global accumulator
    if (threadIdx.x == 0) {
        atomicAdd(sum_out, shmem[0]);
    }
}

// Host function to launch the kernel

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(), "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Determine grid size ensuring we don't oversubscribe
    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size = (grid_size < 1024) ? grid_size : 1024;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "stride_mse_cuda", ([&] {
        stride_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements);
    }));

    // Compute the mean squared error
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Stride Loop Reduction MSE Forward (CUDA)");
}
