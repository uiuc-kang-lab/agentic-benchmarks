#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 128;  // Reduced block size
static const int WARP_SIZE = 32;

template <typename scalar_t>
__global__ void mse_forward_kernel(
    const scalar_t* __restrict__ preds,
    const scalar_t* __restrict__ tgts,
    double* __restrict__ sum_out,
    const int64_t num_elements
) {
    __shared__ double shm[BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int lane = tid % WARP_SIZE;
    int idx = blockIdx.x * blockDim.x + tid;
    
    double thread_sum = 0.0;
    
    // Process multiple elements per thread with grid-stride loop
    while (idx < num_elements) {
        double diff = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        thread_sum += diff * diff;
        idx += blockDim.x * gridDim.x;
    }
    
    // Store in shared memory
    shm[tid] = thread_sum;
    __syncthreads();

    // Two-level reduction: first at warp level (no sync needed), then across warps
    if (tid < 64) {
        shm[tid] += shm[tid + 64];
    }
    __syncthreads();
    
    if (tid < 32) {
        volatile double* vshm = shm;
        vshm[tid] += vshm[tid + 32];
        vshm[tid] += vshm[tid + 16];
        vshm[tid] += vshm[tid + 8];
        vshm[tid] += vshm[tid + 4];
        vshm[tid] += vshm[tid + 2];
        vshm[tid] += vshm[tid + 1];
    }

    if (tid == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.is_cuda(), "predictions must be a CUDA tensor");
    TORCH_CHECK(targets.is_cuda(), "targets must be a CUDA tensor");
    TORCH_CHECK(predictions.numel() == targets.numel(),
                "predictions and targets must have the same number of elements");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Adjust grid size for smaller block size
    const int grid_size = std::min(1024, (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE);

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    auto result = accumulator.div_(static_cast<double>(num_elements));
    return result.to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "Mean Squared Error (MSE) forward (CUDA)");
}