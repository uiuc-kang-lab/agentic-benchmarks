#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size for the kernel
static const int BLOCK_SIZE = 256;

// CUDA kernel that computes the Mean Squared Error using warp-level reduction
// and eliminates shared memory usage for intra-warp reductions.

template <typename scalar_t>
__global__ void warp_atomic_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double thread_sum = 0.0;

    // Calculate global thread index and grid stride
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop: each thread accumulates its portion of the sum of squared differences
    for (; idx < num_elements; idx += stride) {
        double pred_val = static_cast<double>(preds[idx]);
        double tgt_val = static_cast<double>(tgts[idx]);
        double diff = pred_val - tgt_val;
        thread_sum += diff * diff;
    }

    // Perform warp-level reduction using shuffle instructions
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize/2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Each warp's leader (lane 0) atomically accumulates the result
    if ((threadIdx.x & (warpSize - 1)) == 0) {
        atomicAdd(sum_out, thread_sum);
    }
}

// Host function (exposed to Python) that launches the kernel and computes the final mean squared error

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Allocate accumulator as a double tensor for precision
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Determine grid size, limiting maximum grid size to avoid oversubscription
    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size = min(grid_size, 1024);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "warp_atomic_mse_cuda", ([&] {
        warp_atomic_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Compute the mean by dividing the sum of squared errors by the total number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA) using warp-level primitives");
}
