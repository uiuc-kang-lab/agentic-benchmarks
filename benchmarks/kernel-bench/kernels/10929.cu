#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

// Kernel to compute MSE Loss with balanced workload
// Ensures workload is evenly distributed among threads

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE];
    double thread_sum = 0.0;

    // Calculate total number of elements each thread will handle
    // Balance workloads by distributing elements evenly
    int total_threads = blockDim.x * gridDim.x;
    int elements_per_thread = (num_elements + total_threads - 1) / total_threads;

    int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int start_idx = global_thread_id * elements_per_thread;
    int end_idx = min(start_idx + elements_per_thread, num_elements);

    // Process assigned elements
    for (int idx = start_idx; idx < end_idx; ++idx) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
    }

    // Store in shared memory
    shm[threadIdx.x] = thread_sum;
    __syncthreads();

    // Reduction in shared memory
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }

    // Global reduction
    if (threadIdx.x == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

// Forward function

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Ensure grid size is a power of two for optimal reduction
    const int grid_size = (num_elements + BLOCK_SIZE * BLOCK_SIZE - 1) / (BLOCK_SIZE * BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    // Final mean calculation
    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA)");
}
