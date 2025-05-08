#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size to ensure enough threads per block
static const int BLOCK_SIZE = 256;

// CUDA kernel that uses __ldg for aligned global memory accesses and warp shuffle reduction
// to ensure coalesced memory accesses and efficient reduction

template <typename scalar_t>
__global__ void coalesced_shfl_mse_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double thread_sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Grid-stride loop with __ldg to load data from global memory in a coalesced manner
    while (idx < num_elements) {
        // Using __ldg to ensure the read is cached and aligned
        double pred_val = static_cast<double>(__ldg(&preds[idx]));
        double tgt_val = static_cast<double>(__ldg(&tgts[idx]));
        double diff = pred_val - tgt_val;
        thread_sum += diff * diff;
        idx += stride;
    }

    // Warp-level reduction using shuffle instructions for efficient intra-warp communication
    unsigned int mask = 0xffffffff;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(mask, thread_sum, offset);
    }

    // Each warp’s lane 0 writes its sum into shared memory
    __shared__ double shared_sums[32]; // maximum 32 warps per block (256 / 32 = 8, but allocate fixed size)
    int lane = threadIdx.x & (warpSize - 1);
    int warp_id = threadIdx.x / warpSize;
    if (lane == 0) {
        shared_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction across warps in the block
    double block_sum = 0.0;
    int num_warps = blockDim.x / warpSize;
    if (threadIdx.x < num_warps) {
        block_sum = shared_sums[threadIdx.x];
    }
    if (threadIdx.x < warpSize) {
        for (int offset = num_warps / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(mask, block_sum, offset);
        }
    }

    // The first thread in the block atomically accumulates the result
    if (threadIdx.x == 0) {
        atomicAdd(sum_out, block_sum);
    }
}

// Host function invoked from Python via Pybind11
// It sets up the kernel launch parameters, dispatches the kernel, and computes the final mean squared error

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // Limit grid size to avoid oversubscription
    grid_size = min(grid_size, 1024);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "coalesced_shfl_mse_cuda", ([&] {
        coalesced_shfl_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Final mean squared error is the accumulated sum divided by the total number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Coalesced MSE with Warp Shuffle Reduction (CUDA)");
}
