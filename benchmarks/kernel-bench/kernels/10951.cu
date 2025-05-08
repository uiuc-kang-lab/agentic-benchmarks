#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

template <typename scalar_t>
__global__ void mse_forward_kernel_stride(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double thread_sum = 0.0;

    // Strided loop to accumulate squared differences
    for (int i = idx; i < num_elements; i += blockDim.x * gridDim.x) {
        double diff = static_cast<double>(preds[i]) - static_cast<double>(tgts[i]);
        thread_sum += diff * diff;
    }

    // Store partial sums in shared memory
    shm[threadIdx.x] = thread_sum;
    __syncthreads();

    // Block-level reduction
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Add the reduced sum from this block into global accumulator
    if (threadIdx.x == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

torch::Tensor forward_stride(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Use double for accumulation
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    const int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_stride_cuda", [&] {
        mse_forward_kernel_stride<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    // Final mean = accumulator / N
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_stride", &forward_stride, "Mean Squared Error (MSE) forward with stride (CUDA)");
}