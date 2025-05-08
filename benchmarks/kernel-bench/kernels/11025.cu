#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Define block size and warp size
static const int BLOCK_SIZE = 256;
#define WARP_SIZE 32

// CUDA kernel using warp-level reduction to minimize __syncthreads() usage
template <typename scalar_t>
__global__ void mse_forward_kernel_warp_reduction(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double local_sum = 0.0;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // Grid-stride loop to accumulate squared differences
    for (; idx < num_elements; idx += stride) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        local_sum += diff * diff;
    }

    // Warp-level reduction using shuffle intrinsics
    unsigned mask = 0xffffffff;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
    }

    // Each warp writes its reduced result to shared memory
    __shared__ double warp_sums[BLOCK_SIZE / WARP_SIZE];
    int lane = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_sums[warpId] = local_sum;
    }

    // Synchronize once to ensure all warp results are written
    __syncthreads();

    // First thread accumulates results from all warps in the block
    if (threadIdx.x == 0) {
        double block_sum = 0.0;
        int numWarps = blockDim.x / WARP_SIZE;
        for (int i = 0; i < numWarps; i++) {
            block_sum += warp_sums[i];
        }
        atomicAdd(sum_out, block_sum);
    }
}

// Host function that sets up the kernel launch
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();

    // Get SM count and set grid size (using 4 blocks per SM)
    int device_id;
    cudaGetDevice(&device_id);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    int grid_size = sm_count * 4;

    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_warp", ([&]() {
        mse_forward_kernel_warp_reduction<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Final result: mean squared error
    auto result = accumulator.div(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward using warp-level reduction (CUDA)");
}
