#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;
static const int ELEMENTS_PER_THREAD = 4;  // Process multiple elements per thread

template <typename scalar_t>
__global__ void mse_forward_kernel_hybrid(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double thread_sum = 0.0;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int total_threads = stride;
    
    // Process multiple elements per thread for better arithmetic intensity
    for (int i = tid; i < num_elements; i += stride) {
        #pragma unroll
        for (int j = 0; j < ELEMENTS_PER_THREAD && i + j * total_threads < num_elements; j++) {
            int idx = i + j * total_threads;
            double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
            thread_sum += diff * diff;
        }
    }

    // Warp-level reduction using shuffle operations
    unsigned int full_mask = 0xffffffff;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(full_mask, thread_sum, offset);
    }

    // Write warp results to shared memory
    __shared__ double warp_sums[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;

    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    // Final reduction by first warp only
    if (threadIdx.x < (BLOCK_SIZE / 32)) {
        double block_sum = warp_sums[threadIdx.x];
        #pragma unroll
        for (int offset = (BLOCK_SIZE / 32) / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(full_mask, block_sum, offset);
        }

        if (threadIdx.x == 0) {
            atomicAdd(sum_out, block_sum);
        }
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Adjust grid size to account for elements_per_thread
    const int grid_size = std::min(
        (num_elements + (BLOCK_SIZE * ELEMENTS_PER_THREAD) - 1) / (BLOCK_SIZE * ELEMENTS_PER_THREAD),
        65535  // Maximum grid size limit
    );

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_hybrid", ([&] {
        mse_forward_kernel_hybrid<scalar_t><<<grid_size, BLOCK_SIZE>>>(
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
    m.def("forward", &forward, "Hybrid Optimized MSE forward (CUDA)");
}