#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

// Kernel function to compute MSE Loss
template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double thread_sum = 0.0;

    // Strided loop to accumulate squared differences
    while (idx < num_elements) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
        idx += blockDim.x * gridDim.x;
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

// Function to perform forward pass
torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    // Use double for accumulation
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    const int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Create CUDA stream for overlapping operations
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE, 0, stream>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    // Synchronize to ensure kernel execution completion
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Final mean = accumulator / N
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA)");
}
