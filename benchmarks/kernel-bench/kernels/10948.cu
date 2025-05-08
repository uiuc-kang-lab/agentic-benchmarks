#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>

static const int BLOCK_SIZE = 256;

// Device function to calculate squared differences
__device__ double compute_squared_difference(double pred, double tgt) {
    double diff = pred - tgt;
    return diff * diff;
}

// Device function to perform warp-level reduction
__device__ double warp_reduce_sum(double val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Device function to perform block-level reduction
__device__ double block_reduce_sum(double* smem, int tid) {
    int lane = tid % warpSize;
    int warp_id = tid / warpSize;

    // Perform reduction within the warp
    double val = warp_reduce_sum(smem[tid]);

    // Write reduced value to shared memory
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    // Read from shared memory only if we have enough warps
    val = (tid < blockDim.x / warpSize) ? smem[lane] : 0;

    // Final reduction across warps
    if (warp_id == 0) val = warp_reduce_sum(val);

    return val;
}

// Kernel function that calls device functions for modular operations
template <typename scalar_t>
__global__ void modular_mse_kernel(
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

    // Grid-stride loop using device function
    while (idx < num_elements) {
        double pred_val = static_cast<double>(preds[idx]);
        double tgt_val = static_cast<double>(tgts[idx]);
        thread_sum += compute_squared_difference(pred_val, tgt_val);
        idx += grid_stride;
    }

    smem[tid] = thread_sum;
    __syncthreads();

    // Block-level reduction using device function
    double block_sum = block_reduce_sum(smem, tid);

    // Atomic addition to global sum
    if (tid == 0) {
        atomicAdd(sum_out, block_sum);
    }
}

// Host function to launch the kernel
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    grid_size = std::min(grid_size, 1024);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "modular_mse_cuda", ([&] {
        modular_mse_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
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
    m.def("forward", &forward, "Modular MSE forward (CUDA)");
}
