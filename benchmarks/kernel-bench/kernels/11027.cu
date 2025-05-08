#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 256;

// Kernel with optimized synchronization
template <typename scalar_t>
__global__ void mse_forward_kernel_sync_optimized(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    int stride = blockDim.x * gridDim.x;

    double local_sum = 0.0;
    for (int idx = gid; idx < num_elements; idx += stride) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]); if (isnan(diff)) return;
        local_sum += diff * diff;
    }

    shm[tid] = local_sum;
    __syncthreads();  // Ensure all local sums are written to shared memory

    // Reduction in shared memory
    for (int offset = BLOCK_SIZE / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            shm[tid] += shm[tid + offset];
        }
        if (offset > 32) __syncthreads();  // Only sync when necessary
    }

    if (tid == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(), "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    int device_id;
    cudaGetDevice(&device_id);
    int sm_count;
    cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id);
    int grid_size = sm_count * 4;

    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", ([&] {
        mse_forward_kernel_sync_optimized<scalar_t><<<grid_size, BLOCK_SIZE>>>(
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
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA)");
}