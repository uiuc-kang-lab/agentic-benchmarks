#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size for good occupancy; BLOCK_SIZE should be a multiple of warpSize
static const int BLOCK_SIZE = 256;

// This kernel eliminates shared memory for reductions by using warp-level primitives only.
// Each thread accumulates its local sum in a grid-stride loop. Then, within each warp, a reduction is performed
// using __shfl_down_sync, and the warp’s leader atomically adds its partial sum to the global accumulator.

template <typename scalar_t>
__global__ void warp_only_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double local_sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop to cover all elements in the input arrays
    while (idx < num_elements) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        local_sum += diff * diff;
        idx += stride;
    }

    // Warp-level reduction using __shfl_down_sync to sum the values within each warp
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Each warp's leader (first lane in the warp) adds its partial sum directly to the global accumulator
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(sum_out, local_sum);
    }
}

// The host function that launches the CUDA kernel
// and computes the final Mean Squared Error loss

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Use double for accumulation to maintain precision
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Optionally cap the grid size to avoid excessive kernel launches
    grid_size = min(grid_size, 1024);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "warp_only_mse_cuda", ([&] {
        warp_only_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Final MSE: divide the accumulated sum by the total number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE loss computed using warp-level reduction without shared memory (CUDA)");
}
