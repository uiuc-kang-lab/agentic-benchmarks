#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

// Kernel using grid-stride loops and warp shuffle reduction for improved performance
template <typename scalar_t>
__global__ void mse_forward_kernel_stride(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double local_sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Use grid-stride loop to cover all elements with proper boundary check
    for (; idx < num_elements; idx += stride) {
        double pred_val = static_cast<double>(__ldg(&preds[idx]));
        double tgt_val = static_cast<double>(__ldg(&tgts[idx]));
        double diff = pred_val - tgt_val;
        local_sum += diff * diff;
    }

    // Warp-level reduction using shuffle intrinsics
    // Each thread reduces with its warp
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }

    // Each warp's lane 0 holds the sum for that warp
    __shared__ double warp_sums[BLOCK_SIZE / 32];
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        warp_sums[warp_id] = local_sum;
    }

    __syncthreads();

    // First warp aggregates the per-warp results
    if (threadIdx.x < (blockDim.x / 32)) {
        double sum = warp_sums[threadIdx.x];
        for (int offset = (blockDim.x / 32) / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
        if (threadIdx.x == 0) {
            atomicAdd(sum_out, sum);
        }
    }
}

// The PyTorch binding, performing necessary checks and kernel launch
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Accumulate loss in double precision
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_stride", ([&] {
        mse_forward_kernel_stride<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Compute final mean squared error
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA) with stride loops and warp-level reduction");
}
