#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

// CUDA kernel that uses __ldg for aligned global memory accesses and aligns to 128-bit boundaries

template <typename scalar_t>
__global__ void aligned_128bit_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double thread_sum = 0.0;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4; // Align to 128-bit (16 bytes)
    int stride = blockDim.x * gridDim.x * 4;

    // Load data in an efficient, coalesced manner ensuring 128-bit alignment
    using LoadType = double4;

    while (idx < num_elements) {
        if (idx + 3 < num_elements) { // Ensure not to read out of bounds
            double4 pred_vals = __ldg(reinterpret_cast<const double4*>(&preds[idx]));
            double4 tgt_vals = __ldg(reinterpret_cast<const double4*>(&tgts[idx]));
            thread_sum += (pred_vals.x - tgt_vals.x) * (pred_vals.x - tgt_vals.x);
            thread_sum += (pred_vals.y - tgt_vals.y) * (pred_vals.y - tgt_vals.y);
            thread_sum += (pred_vals.z - tgt_vals.z) * (pred_vals.z - tgt_vals.z);
            thread_sum += (pred_vals.w - tgt_vals.w) * (pred_vals.w - tgt_vals.w);
        }
        idx += stride;
    }

    // Warp-level shuffle reduction
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    __shared__ double shared_sums[32];
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) shared_sums[warp_id] = thread_sum;
    __syncthreads();

    // Final reduction across warps
    double block_sum = 0.0;
    int num_warps = blockDim.x / warpSize;
    if (threadIdx.x < num_warps) {
        block_sum = shared_sums[threadIdx.x];
    }
    if (threadIdx.x < warpSize) {
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum_out, block_sum);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);
    // Limit grid size to avoid oversubscription
    grid_size = min(grid_size, 1024);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "aligned_128bit_mse_cuda", ([&] {
        aligned_128bit_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
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
    m.def("forward", &forward, "Aligned 128-bit MSE with __ldg (CUDA)");
}