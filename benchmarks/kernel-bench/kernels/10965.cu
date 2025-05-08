#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void mse_forward_kernel_unroll(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    double thread_sum = 0.0;
    int stride = blockDim.x * gridDim.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Use a for-loop with pragma unroll to reduce loop overhead
    #pragma unroll
    for (int i = idx; i < num_elements; i += stride) {
        double diff = static_cast<double>(preds[i]) - static_cast<double>(tgts[i]);
        thread_sum += diff * diff;
    }

    // Warp-level reduction using __shfl_down_sync with unrolling
    unsigned int full_mask = 0xffffffff;
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(full_mask, thread_sum, offset);
    }

    // Each warp leader writes its result to shared memory
    __shared__ double warp_sums[BLOCK_SIZE / 32 + 1];
    int lane = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp_id] = thread_sum;
    }
    __syncthreads();

    double block_sum = 0.0;
    // First warp loads the partial sums and reduces them
    if (threadIdx.x < (BLOCK_SIZE / 32)) {
        block_sum = warp_sums[threadIdx.x];
        #pragma unroll
        for (int offset = (BLOCK_SIZE / 32) / 2; offset > 0; offset /= 2) {
            block_sum += __shfl_down_sync(full_mask, block_sum, offset);
        }
    }

    if (threadIdx.x == 0) {
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
    // Accumulator in double precision for numerical accuracy
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda_unroll", ([&] {
        mse_forward_kernel_unroll<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    }));

    // Compute final mean: sum divided by number of elements
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Optimized Mean Squared Error (MSE) forward (CUDA) kernel with loop unrolling");
}
