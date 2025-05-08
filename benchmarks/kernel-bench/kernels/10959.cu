#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size
static const int BLOCK_SIZE = 256;

// CUDA kernel using warp-level primitives for reduction
// Each thread performs a grid-stride loop to compute its partial sum of squared differences.
// Then, within the warp, __shfl_down_sync is used to reduce the sum to the warp leader.
// Finally, each warp leader (lane 0) issues an atomicAdd to the global accumulator in double precision.

template <typename scalar_t>
__global__ void mse_forward_kernel_warp_atomic(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double thread_sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    // Grid-stride loop to compute partial sum
    while (idx < num_elements) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
        idx += stride;
    }

    // Use warp-level reduction with __shfl_down_sync to sum values within a warp
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Each warp's leader (lane 0) performs an atomicAdd
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(sum_out, thread_sum);
    }
}

// Host function to launch the kernel

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Accumulate in double precision for correctness
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    const int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_warp_atomic_cuda", ([&] {
        mse_forward_kernel_warp_atomic<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Compute mean squared error by dividing by the number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA) using warp-level primitives");
}
