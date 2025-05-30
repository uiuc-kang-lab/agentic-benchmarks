#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

static const int BLOCK_SIZE = 512;

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

    // Use registers for temporary accumulation to reduce shared memory access
    // Process multiple elements per thread to increase arithmetic intensity
    while (idx < num_elements) {
        // Load multiple elements at once to improve memory bandwidth utilization
        double diff1 = 0.0, diff2 = 0.0;
        if (idx < num_elements) {
            diff1 = static_cast<double>(preds[idx]) - static_cast<double>(tgts[idx]);
        }
        if (idx + blockDim.x * gridDim.x < num_elements) {
            diff2 = static_cast<double>(preds[idx + blockDim.x * gridDim.x]) 
                   - static_cast<double>(tgts[idx + blockDim.x * gridDim.x]);
        }
        
        thread_sum += diff1 * diff1 + diff2 * diff2;
        idx += 2 * blockDim.x * gridDim.x;
    }

    // Store final result in shared memory, avoiding bank conflicts
    shm[threadIdx.x] = thread_sum;
    __syncthreads();

    // Simplified reduction for power-of-two block size
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            shm[threadIdx.x] += shm[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(sum_out, shm[0]);
    }
}

torch::Tensor forward(torch::Tensor predictions, torch::Tensor targets) {
    TORCH_CHECK(predictions.device().is_cuda() && targets.device().is_cuda(),
               "Inputs must be CUDA tensors");
    TORCH_CHECK(predictions.numel() == targets.numel(),
               "Input sizes must match");

    const int64_t num_elements = predictions.numel();
    auto accumulator = torch::zeros({1}, predictions.options().dtype(at::kDouble));

    // Increased occupancy with 512-thread blocks
    const int grid_size = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES(predictions.scalar_type(), "mse_forward_cuda", [&] {
        mse_forward_kernel<scalar_t><<<grid_size, BLOCK_SIZE>>>(
            predictions.data_ptr<scalar_t>(),
            targets.data_ptr<scalar_t>(),
            accumulator.data_ptr<double>(),
            num_elements
        );
    });

    return accumulator.div_(num_elements).to(predictions.dtype());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "MSE forward (CUDA optimized block size");
}